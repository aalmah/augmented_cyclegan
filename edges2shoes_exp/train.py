#!/usr/bin/env python

import sys
import os
from shutil import copyfile
import time
import itertools
import torch
from torch.autograd import Variable
import torchvision.utils as vutils

from options import TrainOptions, create_sub_dirs
from edges2shoes_data import DataLoader, load_edges2shoes, AlignedIterator, UnalignedIterator
from model import StochCycleGAN, AugmentedCycleGAN
import numpy as np
from evaluate import eval_mse_A, eval_ubo_B
import shutil
import random
import glob
import json

def save_results(expr_dir, results_dict):
    # save to results.json (for cluster exp)
    fname = os.path.join(expr_dir, 'results.json')
    with open(fname, 'w') as f:
        json.dump(results_dict, f, indent=4)


def copy_scripts_to_folder(expr_dir):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for f in glob.glob("%s/*.py" % dir_path):
        shutil.copy(f, expr_dir)

def print_log(out_f, message):
    out_f.write(message+"\n")
    out_f.flush()
    print(message)

def format_log(epoch, i, errors, t, prefix=True):
    message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
    if not prefix:
        message = ' ' * len(message)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)
    return message

def visualize_cycle(opt, real_A, visuals, eidx, uidx, train):
    size = real_A.size()

    images = [img.cpu().unsqueeze(1) for img in visuals.values()]
    vis_image = torch.cat(images, dim=1).view(size[0]*len(images),size[1],size[2],size[3])
    if train:
        save_path = opt.train_vis_cycle
    else:
        save_path = opt.vis_cycle
    save_path = os.path.join(save_path, 'cycle_%02d_%04d.png' % (eidx, uidx))
    vutils.save_image(vis_image.cpu(), save_path,
        normalize=True, range=(-1,1), nrow=len(images))
    copyfile(save_path, os.path.join(opt.vis_latest, 'cycle.png'))

def visualize_multi(opt, real_A, model, eidx, uidx):
    size = real_A.size()
    # all samples in real_A share the same prior_z_B
    multi_prior_z_B = Variable(real_A.data.new(opt.num_multi,
        opt.nlatent, 1, 1).normal_(0, 1).repeat(size[0],1,1,1), volatile=True)
    multi_fake_B = model.generate_multi(real_A.detach(), multi_prior_z_B)
    multi_fake_B = multi_fake_B.data.cpu().view(
        size[0], opt.num_multi, size[1], size[2], size[3])
    vis_multi_image = torch.cat([real_A.data.cpu().unsqueeze(1), multi_fake_B], dim=1) \
        .view(size[0]*(opt.num_multi+1),size[1],size[2],size[3])
    save_path = os.path.join(opt.vis_multi, 'multi_%02d_%04d.png' % (eidx, uidx))
    vutils.save_image(vis_multi_image.cpu(), save_path,
        normalize=True, range=(-1,1), nrow=opt.num_multi+1)
    copyfile(save_path, os.path.join(opt.vis_latest, 'multi.png'))

def visualize_inference(opt, real_A, real_B, model, eidx, uidx):
    size = real_A.size()

    real_B = real_B[:opt.num_multi]
    # all samples in real_A share the same post_z_B
    multi_fake_B = model.inference_multi(real_A.detach(), real_B.detach())
    multi_fake_B = multi_fake_B.data.cpu().view(
        size[0], opt.num_multi, size[1], size[2], size[3])

    vis_multi_image = torch.cat([real_A.data.cpu().unsqueeze(1), multi_fake_B], dim=1) \
        .view(size[0]*(opt.num_multi+1),size[1],size[2],size[3])

    vis_multi_image = torch.cat([torch.ones(1, size[1], size[2], size[3]).cpu(), real_B.data.cpu(),
                                 vis_multi_image.cpu()], dim=0)

    save_path = os.path.join(opt.vis_inf, 'inf_%02d_%04d.png' % (eidx, uidx))
    vutils.save_image(vis_multi_image.cpu(), save_path,
        normalize=True, range=(-1,1), nrow=opt.num_multi+1)
    copyfile(save_path, os.path.join(opt.vis_latest, 'inf.png'))

def train_model():
    opt = TrainOptions().parse(sub_dirs=['vis_multi','vis_cycle','vis_latest','train_vis_cycle'])
    out_f = open("%s/results.txt" % opt.expr_dir, 'w')
    copy_scripts_to_folder(opt.expr_dir)
    use_gpu = len(opt.gpu_ids) > 0

    if opt.seed is not None:
        print "using random seed:", opt.seed
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        if use_gpu:
            torch.cuda.manual_seed_all(opt.seed)

    if opt.numpy_data:
        trainA, trainB, devA, devB, testA, testB = load_edges2shoes(opt.dataroot)
        train_dataset = UnalignedIterator(trainA, trainB, batch_size=opt.batchSize)
        print_log(out_f, '#training images = %d' % len(train_dataset))
        vis_inf = False

        test_dataset = AlignedIterator(testA, testB, batch_size=100)
        print_log(out_f, '#test images = %d' % len(test_dataset))

        dev_dataset = AlignedIterator(devA, devB, batch_size=100)
        print_log(out_f, '#dev images = %d' % len(dev_dataset))

        dev_cycle = itertools.cycle(AlignedIterator(devA, devB, batch_size=25))
    else:
        train_data_loader = DataLoader(opt, subset='train', unaligned=True, batchSize=opt.batchSize)
        test_data_loader = DataLoader(opt, subset='val', unaligned=False, batchSize=200)
        dev_data_loader = DataLoader(opt, subset='dev', unaligned=False, batchSize=200)
        dev_cycle_loader = DataLoader(opt, subset='dev', unaligned=False, batchSize=25)

        train_dataset = train_data_loader.load_data()
        dataset_size = len(train_data_loader)
        print_log(out_f, '#training images = %d' % dataset_size)
        vis_inf = False

        test_dataset = test_data_loader.load_data()
        print_log(out_f, '#test images = %d' % len(test_data_loader))

        dev_dataset = dev_data_loader.load_data()
        print_log(out_f, '#dev images = %d' % len(dev_data_loader))

        dev_cycle = itertools.cycle(dev_cycle_loader.load_data())

    if opt.supervised:
        if opt.numpy_data:
            sup_size = int(len(trainA) * opt.sup_frac)
            sup_trainA = trainA[:sup_size]
            sup_trainB = trainB[:sup_size]
            sup_train_dataset = AlignedIterator(sup_trainA, sup_trainB, batch_size=opt.batchSize)
        else:
            sup_train_data_loader = DataLoader(opt, subset='train', unaligned=False,
                                               batchSize=opt.batchSize, fraction=opt.sup_frac)
            sup_train_dataset = sup_train_data_loader.load_data()
            sup_size = len(sup_train_data_loader)
        sup_train_dataset = itertools.cycle(sup_train_dataset)
        print_log(out_f, '#supervised images = %d' % sup_size)

    # create_model
    if opt.model == 'stoch_cycle_gan':
        model = StochCycleGAN(opt)
    elif opt.model == 'cycle_gan':
        model = StochCycleGAN(opt, ignore_noise=True)
    elif opt.model == 'aug_cycle_gan':
        model = AugmentedCycleGAN(opt)
        create_sub_dirs(opt, ['vis_inf'])
        vis_inf = True
    else:
        raise NotImplementedError('Specified model is not implemented.')

    print_log(out_f, "model [%s] was created" % (model.__class__.__name__))

    # visualizer = Visualizer(opt)
    total_steps = 0
    print_start_time = time.time()
    results = {
        'best_dev_mse_A'  : sys.float_info.max,
        'best_test_mse_A' : sys.float_info.max,
        'best_dev_bpp_B'  : sys.float_info.max,
        'best_test_bpp_B' : sys.float_info.max,
    }
    save_results(opt.expr_dir, results)
    history_mse_A = []
    history_ubo_B = []

    create_sub_dirs(opt, ['vis_pred_B'])

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_dataset):
            real_A, real_B = Variable(data['A']), Variable(data['B'])
            if real_A.size(0) != real_B.size(0):
                continue
            prior_z_B = Variable(real_A.data.new(real_A.size(0), opt.nlatent, 1, 1).normal_(0, 1))

            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            if use_gpu:
                real_A = real_A.cuda()
                real_B = real_B.cuda()
                prior_z_B = prior_z_B.cuda()

            if opt.monitor_gnorm:
                losses, visuals, gnorms = model.train_instance(real_A, real_B, prior_z_B)
            else:
                losses, visuals = model.train_instance(real_A, real_B, prior_z_B)

            # supervised training
            if opt.supervised:
                sup_data = sup_train_dataset.next()
                sup_real_A, sup_real_B = Variable(sup_data['A']), Variable(sup_data['B'])
                if use_gpu:
                    sup_real_A, sup_real_B = sup_real_A.cuda(), sup_real_B.cuda()
                sup_losses = model.supervised_train_instance(sup_real_A, sup_real_B, prior_z_B)

            if total_steps % opt.display_freq == 0:

                # visualize current training batch
                visualize_cycle(opt, real_A, visuals, epoch, epoch_iter/opt.batchSize, train=True)

                dev_data = dev_cycle.next()
                dev_real_A, dev_real_B = Variable(dev_data['A']), Variable(dev_data['B'])
                dev_prior_z_B = Variable(dev_real_A.data.new(dev_real_A.size(0),
                                                             opt.nlatent, 1, 1).normal_(0, 1))
                if use_gpu:
                    dev_real_A = dev_real_A.cuda()
                    dev_real_B = dev_real_B.cuda()
                    dev_prior_z_B = dev_prior_z_B.cuda()

                dev_visuals = model.generate_cycle(dev_real_A, dev_real_B, dev_prior_z_B)
                visualize_cycle(opt, dev_real_A, dev_visuals, epoch, epoch_iter/opt.batchSize, train=False)

                # visualize generated B with different z_B
                visualize_multi(opt, dev_real_A, model, epoch, epoch_iter/opt.batchSize)

                if vis_inf:
                    # visualize generated B with different z_B infered from real_B
                    visualize_inference(opt, dev_real_A, dev_real_B, model, epoch, epoch_iter/opt.batchSize)

            if total_steps % opt.print_freq == 0:
                t = (time.time() - print_start_time) / opt.batchSize
                print_log(out_f, format_log(epoch, epoch_iter, losses, t))
                if opt.supervised:
                    print_log(out_f, format_log(epoch, epoch_iter, sup_losses, t, prefix=False))
                if opt.monitor_gnorm:
                    print_log(out_f, format_log(epoch, epoch_iter, gnorms, t, prefix=False)+"\n")
                print_start_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print_log(out_f, 'saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
            model.save('latest')

        #####################
        # evaluate mappings
        #####################
        if epoch % opt.eval_A_freq == 0:
            t = time.time()
            dev_mse_A = eval_mse_A(dev_dataset, model)
            test_mse_A = eval_mse_A(test_dataset, model)
            t = time.time() - t
            history_mse_A.append((dev_mse_A, test_mse_A))
            np.save("%s/history_mse_A" % opt.expr_dir, history_mse_A)

            res_str_list = ["[%d] DEV_MSE_A: %.4f, TEST_MSE_A: %.4f, TIME: %.4f" % (epoch, dev_mse_A, test_mse_A, t)]
            if dev_mse_A < results['best_dev_mse_A']:
                with open("%s/best_mse_A.txt" % opt.expr_dir, 'w') as best_mse_A_f:
                    best_mse_A_f.write(res_str_list[0]+'\n')
                    best_mse_A_f.flush()
                results['best_dev_mse_A'] = dev_mse_A
                results['best_test_mse_A'] = test_mse_A
                model.save('best_A')
                save_results(opt.expr_dir, results)
                res_str_list += ["*** BEST DEV A ***"]
            res_str = "\n".join(["-"*60] + res_str_list + ["-"*60])
            print_log(out_f, res_str)

        if epoch % opt.eval_B_freq == 0:
            t = time.time()
            if opt.model == 'cycle_gan':
                steps = 1
            else:
                steps = 50
            dev_ubo_B, dev_bpp_B, dev_kld_B = eval_ubo_B(dev_dataset, model, steps, True, 'pred_B_%d' % epoch,
                                                         opt.vis_pred_B)
            test_ubo_B, test_bpp_B, test_kld_B = eval_ubo_B(test_dataset, model, steps, False, 'pred_B',
                                                            opt.vis_pred_B)
            t = time.time() - t
            history_ubo_B.append((dev_ubo_B, dev_bpp_B, dev_kld_B, test_ubo_B, test_bpp_B, test_kld_B))
            np.save("%s/history_ubo_B" % opt.expr_dir, history_ubo_B)

            res_str_list = ["[%d] DEV_BPP_B: %.4f, TEST_BPP_B: %.4f, TIME: %.4f" % (epoch, dev_bpp_B, test_bpp_B, t)]
            if dev_bpp_B < results['best_dev_bpp_B']:
                with open("%s/best_bpp_B.txt" % opt.expr_dir, 'w') as best_bpp_B_f:
                    best_bpp_B_f.write(res_str_list[0]+'\n')
                    best_bpp_B_f.flush()
                results['best_dev_bpp_B'] = dev_bpp_B
                results['best_test_bpp_B'] = test_bpp_B
                save_results(opt.expr_dir, results)
                model.save('best_B')
                res_str_list += ["*** BEST BPP B ***"]
            res_str = "\n".join(["-"*60] + res_str_list + ["-"*60])
            print_log(out_f, res_str)

        print_log(out_f, 'End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if epoch > opt.niter:
            model.update_learning_rate()

    out_f.close()


if __name__ == "__main__":
    train_model()
