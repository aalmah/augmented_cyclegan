import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import math
from model import gauss_reparametrize, log_prob_laplace, log_prob_gaussian, kld_std_guss

def eval_mse_A(dataset, model, use_gpu=True):
    mse_A = []
    for batch in dataset:
        real_A, real_B = Variable(batch['A']), Variable(batch['B'])
        if use_gpu:
            real_A = real_A.cuda()
            real_B = real_B.cuda()
        pred_A = model.predict_A(real_B)
        mse_A.append(F.mse_loss(pred_A, real_A).data[0])
    return np.mean(mse_A)

def eval_ubo_B(dataset, model, steps=500, visualize=False, vis_name=None, vis_path=None,
               verbose=False, logvar_B=None, use_gpu=True, vis_batch=25, compute_l1=False):
    ubo_B = []
    bpp_B = []
    kld_B = []
    for batch in dataset:
        real_A, real_B = Variable(batch['A']), Variable(batch['B'])
        if use_gpu:
            real_A = real_A.cuda()
            real_B = real_B.cuda()
        ubo, kld, bpp = variational_ubo(model, real_A, real_B, steps,
                                        visualize, vis_name, vis_path, verbose, logvar_B,
                                        use_gpu, vis_batch, compute_l1)
        ubo_B.append(ubo)
        bpp_B.append(bpp)
        kld_B.append(kld)
    return np.mean(ubo_B), np.mean(bpp_B), np.mean(kld_B)

def variational_ubo(model, real_A, real_B, steps, visualize=False, vis_name=None, vis_path=None,
                    verbose=False, logvar_B=None, use_gpu=True, vis_batch=25, compute_l1=False):
    if visualize:
        assert vis_name is not None and vis_path is not None
    dequant = Variable(torch.zeros(*real_B.size()).uniform_(0, 1./127.5).cuda())
    size = real_A.size()
    vis_size = real_A[:vis_batch].size()

    # define q params
    mu = Variable(torch.zeros(size[0], model.opt.nlatent).cuda(), requires_grad=True)
    logvar = Variable(torch.zeros(size[0], model.opt.nlatent).fill_(math.log(0.01)).cuda(),
                      requires_grad=True)
    if logvar_B is None:
        logvar_B = Variable(torch.zeros(1, 3, 64, 64).fill_(math.log(0.01)).cuda())
    # fake_A = model.predict_A(real_B)

    # init mu with encoder values
    if hasattr(model, 'netE_B'):
        params = model.predict_enc_params(real_A, real_B)
        enc_mu = params[0]
        mu = Variable(enc_mu.data, requires_grad=True)
        if len(params) == 2:
            enc_logvar = params[1]
            logvar = Variable(enc_logvar.data, requires_grad=True)

    lr = 1e-2
    iterative_opt = torch.optim.RMSprop([mu, logvar], lr=1e-2)

    real_B = real_B + dequant
    rA = Variable(real_A.data, volatile=True)

    z_B = gauss_reparametrize(mu, logvar)
    fake_B = model.predict_B(real_A, z_B)

    if compute_l1:
        if model.opt.stoch_enc:
            rec_B = fake_B
        else:
            x = Variable(mu.view(mu.size(0), mu.size(1), 1, 1).data, volatile=True)
            rec_B = model.predict_B(rA, x)

    if visualize:
        if model.opt.stoch_enc:
            vis_z_B = z_B[:vis_batch]
        else:
            vis_z_B = mu.view(mu.size(0), mu.size(1), 1, 1)[:vis_batch]
        vis_B = model.predict_B(real_A[:vis_batch], vis_z_B)
        save_path = os.path.join(vis_path, '%s_0.png' % vis_name)
        visualize_data(model.opt, [real_A.data[:vis_batch], real_B.data[:vis_batch],
                                   vis_B.data],
                       vis_size, save_path)

    # ubo_val = None
    for i in range(steps):
        # reshape
        z_B = z_B.view(size[0], model.opt.nlatent)

        log_prob = log_prob_laplace(real_B, fake_B, logvar_B)
        log_prob = log_prob.view(size[0], -1).sum(1)

        # log_prob_det = log_prob_gaussian_detail(real_B, fake_B, x_logvar, (size[0], n_sample, -1))

        # kld = log_prob_gaussian(z_B, mu, logvar) -\
        #       log_prob_gaussian(z_B, 0*mu, 0*mu)
        # kld = kld.sum(1)
        kld = kld_std_guss(mu, logvar)

        ubo = (-log_prob + kld) + (64*64*3) * math.log(127.5)
        ubo_val_new = ubo.mean(0).data[0]
        kld_val = kld.mean(0).data[0]
        bpp = ubo.mean(0).data[0] / (64*64*3* math.log(2.))
        if compute_l1:
            l1_loss = F.l1_loss(real_B, rec_B).mean(0).data[0]
        if verbose:
            res_str = '[%d] UBO: %.4f, KLD: %.4f, BPP: %.4f' % (i, ubo_val_new, kld_val, bpp)
            if compute_l1:
                res_str = '%s, L1: %.4f' % (res_str, l1_loss)
            print res_str

        # if ubo_val is not None and abs(ubo_val - ubo_val_new) < 1e-4:
        #     return ubo_val_new, kld_val, bpp

        ubo_val = ubo_val_new
        loss = ubo.mean(0)
        iterative_opt.zero_grad()
        loss.backward()
        iterative_opt.step()

        z_B = gauss_reparametrize(mu, logvar)
        fake_B = model.predict_B(real_A, z_B)
        if compute_l1:
            if model.opt.stoch_enc:
                rec_B = fake_B
            else:
                x = Variable(mu.view(mu.size(0), mu.size(1), 1, 1).data, volatile=True)
                rec_B = model.predict_B(rA, x)

        if visualize and (i+1) % 100 == 0:
            if model.opt.stoch_enc:
                vis_z_B = z_B[:vis_batch]
            else:
                vis_z_B = mu.view(mu.size(0), mu.size(1), 1, 1)[:vis_batch]
            vis_B = model.predict_B(real_A[:vis_batch], vis_z_B)
            save_path = os.path.join(vis_path, '%s_%d.png' % (vis_name, i+1))
            visualize_data(model.opt, [real_A.data[:vis_batch], real_B.data[:vis_batch],
                                       vis_B.data],
                           vis_size, save_path)
            # lr /= 2.
            # for param_group in iterative_opt.param_groups:
            #     param_group['lr'] = lr

    return ubo_val, kld_val, bpp

def nats2bpp(nats, nc, npx):
    bpp = (nats / (npx * npx * nc)) / np.log(2.)
    return bpp


def visualize_data(opt, data, size, save_path):
    images = [img.cpu().unsqueeze(1) for img in data]
    vis_image = torch.cat(images, dim=1).view(size[0]*len(images),size[1],size[2],size[3])
    vutils.save_image(vis_image.cpu(), save_path,
        normalize=True, range=(-1,1), nrow=len(images))
