import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np

USE_PYTORCH_IN = False


######################################################################
# Superclass of all Modules that take two inputs
######################################################################
class TwoInputModule(nn.Module):
    def forward(self, input1, input2):
        raise NotImplementedError

######################################################################
# A (sort of) hacky way to create a module that takes two inputs (e.g. x and z)
# and returns one output (say o) defined as follows:
# o = module2.forward(module1.forward(x), z)
# Note that module2 MUST support two inputs as well.
######################################################################
class MergeModule(TwoInputModule):
    def __init__(self, module1, module2):
        """ module1 could be any module (e.g. Sequential of several modules)
            module2 must accept two inputs
        """
        super(MergeModule, self).__init__()
        self.module1 = module1
        self.module2 = module2

    def forward(self, input1, input2):
        output1 = self.module1.forward(input1)
        output2 = self.module2.forward(output1, input2)
        return output2

######################################################################
# A (sort of) hacky way to create a container that takes two inputs (e.g. x and z)
# and applies a sequence of modules (exactly like nn.Sequential) but MergeModule
# is one of its submodules it applies it to both inputs
######################################################################
class TwoInputSequential(nn.Sequential, TwoInputModule):
    def __init__(self, *args):
        super(TwoInputSequential, self).__init__(*args)

    def forward(self, input1, input2):
        """overloads forward function in parent calss"""

        for module in self._modules.values():
            if isinstance(module, TwoInputModule):
                input1 = module.forward(input1, input2)
            else:
                input1 = module.forward(input1)
        return input1


######################################################################
# A standard instance norm module.
# Since the pytorch instance norm used BatchNorm as a base and thus is 
# different from the standard implementation.
######################################################################
class InstanceNorm(nn.Module):
    def __init__(self, num_features, affine=True, eps=1e-5):
        """`num_features` number of feature channels
        """
        super(InstanceNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        self.scale = Parameter(torch.Tensor(num_features))
        self.shift = Parameter(torch.Tensor(num_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.scale.data.normal_(mean=0., std=0.02)
            self.shift.data.zero_()

    def forward(self, input):
        size = input.size()
        x_reshaped = input.view(size[0], size[1], size[2]*size[3])
        mean = x_reshaped.mean(2, keepdim=True)
        centered_x = x_reshaped - mean
        std = torch.rsqrt((centered_x ** 2).mean(2, keepdim=True) + self.eps)
        norm_features = (centered_x * std).view(*size)

        # broadcast on the batch dimension, hight and width dimensions
        if self.affine:
            output = norm_features * self.scale[:,None,None] + self.shift[:,None,None]
        else:
            output = norm_features

        return output
InstanceNorm2d = nn.InstanceNorm2d if USE_PYTORCH_IN else InstanceNorm

######################################################################
# A module implementing conditional instance norm.
# Takes two inputs: x (input features) and z (latent codes)
######################################################################
class CondInstanceNorm(TwoInputModule):
    def __init__(self, x_dim, z_dim, eps=1e-5):
        """`x_dim` dimensionality of x input
           `z_dim` dimensionality of z latents
        """
        super(CondInstanceNorm, self).__init__()
        self.eps = eps
        self.shift_conv = nn.Sequential(
            nn.Conv2d(z_dim, x_dim, kernel_size=1, padding=0, bias=True),
            nn.ReLU(True)
        )
        self.scale_conv = nn.Sequential(
            nn.Conv2d(z_dim, x_dim, kernel_size=1, padding=0, bias=True),
            nn.ReLU(True)
        )

    def forward(self, input, noise):
        # assumes input.dim() == 4, TODO: generalize that.

        shift = self.shift_conv.forward(noise)
        scale = self.scale_conv.forward(noise)
        size = input.size()
        x_reshaped = input.view(size[0], size[1], size[2]*size[3])
        mean = x_reshaped.mean(2, keepdim=True)
        var = x_reshaped.var(2, keepdim=True)
        std =  torch.rsqrt(var + self.eps)
        norm_features = ((x_reshaped - mean) * std).view(*size)
        output = norm_features * scale + shift
        return output


######################################################################
# A modified resnet block which allows for passing additional noise input
# to be used for conditional instance norm
######################################################################
class CINResnetBlock(TwoInputModule):
    def __init__(self, x_dim, z_dim, padding_type, norm_layer, use_dropout, use_bias):
        super(CINResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(x_dim, z_dim, padding_type, norm_layer, use_dropout, use_bias)
        self.relu = nn.ReLU(True)

        for idx, module in enumerate(self.conv_block):
            self.add_module(str(idx), module)

    def build_conv_block(self, x_dim, z_dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            MergeModule(
                nn.Conv2d(x_dim, x_dim, kernel_size=3, padding=p, bias=use_bias),
                norm_layer(x_dim, z_dim)
            ),
            nn.ReLU(True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(x_dim, x_dim, kernel_size=3, padding=p, bias=use_bias),
                       InstanceNorm2d(x_dim, affine=True)]

        return TwoInputSequential(*conv_block)

    def forward(self, x, noise):
        out = self.conv_block(x, noise)
        out = self.relu(x + out)
        return out

######################################################################
# Define a resnet block
######################################################################
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        self.relu = nn.ReLU(True)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        conv_block += [nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        conv_block += [norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        out = self.relu(x + out)
        return out
