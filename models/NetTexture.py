import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from .base_model import BaseModel
from . import networks
import numpy as np
import functools
from PIL import Image
from util import util
from torchvision import models
from collections import namedtuple

from util import Embedder


class NetTexture(nn.Module):
    def __init__(self, sample_point_dim, texture_features, n_layers=8, n_freq=10, ngf=256):
        super(NetTexture, self).__init__()
        self.ngf = ngf
        self.layers = n_layers
        self.skips=[4]

        multires = n_freq
        self.embedder, self.embedder_out_dim = Embedder.get_embedder(multires, input_dims=sample_point_dim, i=0)

        print('sample_point_dim: ', sample_point_dim)
        print('embedder_out_dim: ', self.embedder_out_dim)

        self.pts_linears = nn.ModuleList(
            [nn.Conv2d(self.embedder_out_dim , self.ngf, kernel_size=1, stride=1, padding=0)] +
            [nn.Conv2d(self.ngf, self.ngf, kernel_size=1, stride=1, padding=0) if i not in self.skips else nn.Conv2d(self.ngf + self.embedder_out_dim , self.ngf, kernel_size=1, stride=1, padding=0) for i in range(self.layers-1)])
        self.output_linear = nn.Conv2d(self.ngf, texture_features, kernel_size=1, stride=1, padding=0)


        self.simple_lin = nn.Conv2d(sample_point_dim, texture_features, kernel_size=1, stride=1, padding=0)



    def forward(self, input_pts):
        simple_net = False
        if simple_net:
            return self.simple_lin(input_pts)
        else:
            # input_pts in [-1,1]
            #input_pts = input_pts * PI # --> [-PI, PI]
            input_embed = self.embedder(input_pts)
            h = input_embed
            for i, l in enumerate(self.pts_linears):
                h = self.pts_linears[i](h)
                #h = F.relu(h)
                h = F.leaky_relu(h)
                if i in self.skips:
                    h = torch.cat([input_embed, h], 1)
            return self.output_linear(h)


    def regularizer(self):
        return 0.0
