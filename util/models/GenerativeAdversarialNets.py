#!/usr/local/bin/python
# coding: utf-8
'''
@University  : FEI
@Local       : São Bernardo do Campo, São Paulo, Brazil
@Laboratory  : Image Processing Lab (IPL)
@file        : GenerativeAdversarialNets.py
@author      : Lucas Fontes Buzuti
@version     : 1.0.0
@created     : 06/01/2020
@modified    : 14/10/2020
@e-mail      : lucas.buzuti@outlook.com
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from util.config import cfg



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """ Generator """
    def __init__(self):
        super(Generator, self).__init__()
        #self.ngpu = ngpu

        # input is Z, going into a convolution
        self.convT1 = nn.ConvTranspose2d(
            cfg.MODEL.Z, cfg.MODEL.FEATURE_MAPS_G*16, 4, 1, 0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_G*16)
        # state size. (ngf*16) x 4 x 4
        self.convT2 = nn.ConvTranspose2d(
            cfg.MODEL.FEATURE_MAPS_G*16, cfg.MODEL.FEATURE_MAPS_G*8, 4, 2, 1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_G*8)
        # state size. (ngf*8) x 8 x 8
        self.convT3 = nn.ConvTranspose2d(
            cfg.MODEL.FEATURE_MAPS_G*8, cfg.MODEL.FEATURE_MAPS_G*4, 4, 2, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_G*4)
        # state size. (ngf*4) x 16 x 16
        self.convT4 = nn.ConvTranspose2d(
            cfg.MODEL.FEATURE_MAPS_G*4, cfg.MODEL.FEATURE_MAPS_G*2, 4, 2, 1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_G*2)
        # state size. (ngf*2) x 32 x 32
        self.convT5 = nn.ConvTranspose2d(
            cfg.MODEL.FEATURE_MAPS_G*2, cfg.MODEL.FEATURE_MAPS_G, 4, 2, 1, bias=False
        )
        self.bn5 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_G)
        # state size. (ngf) x 64 x 64
        self.convT6 = nn.ConvTranspose2d(
            cfg.MODEL.FEATURE_MAPS_G, cfg.DATASET.CHANNEL, 4, 2, 1, bias=False
        )
        # state size. (nc) x 128 x 128

    def forward(self, x):
        x = self.convT1(x)
        x = F.relu(self.bn1(x), True)
        x = self.convT2(x)
        x = F.relu(self.bn2(x), True)
        x = self.convT3(x)
        x = F.relu(self.bn3(x), True)
        x = self.convT4(x)
        x = F.relu(self.bn4(x), True)
        x = self.convT5(x)
        x = F.relu(self.bn5(x), True)
        x = self.convT6(x)
        return torch.tanh(x)


class Generator_v2(nn.Module):
    """ Generator """
    def __init__(self):
        super(Generator_v2, self).__init__()
        #self.ngpu = ngpu

        # input is Z, going into a convolution
        self.convT1 = nn.ConvTranspose2d(
            cfg.MODEL.Z, cfg.MODEL.FEATURE_MAPS_G*64, 4, 1, 0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_G*64)
        # state size. (ngf*64) x 4 x 4
        self.convT2 = nn.ConvTranspose2d(
            cfg.MODEL.FEATURE_MAPS_G*64, cfg.MODEL.FEATURE_MAPS_G*32, 4, 2, 1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_G*32)
        # state size. (ngf*32) x 8 x 8
        self.convT3 = nn.ConvTranspose2d(
            cfg.MODEL.FEATURE_MAPS_G*32, cfg.MODEL.FEATURE_MAPS_G*16, 4, 2, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_G*16)
        # state size. (ngf*16) x 16 x 16
        self.convT4 = nn.ConvTranspose2d(
            cfg.MODEL.FEATURE_MAPS_G*16, cfg.MODEL.FEATURE_MAPS_G*8, 4, 2, 1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_G*8)
        # state size. (ngf*8) x 32 x 32
        self.convT5 = nn.ConvTranspose2d(
            cfg.MODEL.FEATURE_MAPS_G*8, cfg.MODEL.FEATURE_MAPS_G*4, 4, 2, 1, bias=False
        )
        self.bn5 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_G*4)
        # state size. (ngf*4) x 64 x 64
        self.convT6 = nn.ConvTranspose2d(
            cfg.MODEL.FEATURE_MAPS_G*4, cfg.MODEL.FEATURE_MAPS_G*2, 4, 2, 1, bias=False
        )
        self.bn6 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_G*2)
        # state size. (ngf*2) x 128 x 128
        self.convT7 = nn.ConvTranspose2d(
            cfg.MODEL.FEATURE_MAPS_G*2, cfg.MODEL.FEATURE_MAPS_G, 4, 2, 1, bias=False
        )
        self.bn7 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_G)
        # state size. (ngf) x 256 x 256
        self.convT8 = nn.ConvTranspose2d(
            cfg.MODEL.FEATURE_MAPS_G, cfg.DATASET.CHANNEL, 4, 2, 1, bias=False
        )
        # state size. (nc) x 512 x 512

    def forward(self, x):
        x = self.convT1(x)
        x = F.relu(self.bn1(x), True)
        x = self.convT2(x)
        x = F.relu(self.bn2(x), True)
        x = self.convT3(x)
        x = F.relu(self.bn3(x), True)
        x = self.convT4(x)
        x = F.relu(self.bn4(x), True)
        x = self.convT5(x)
        x = F.relu(self.bn5(x), True)
        x = self.convT6(x)
        x = F.relu(self.bn6(x), True)
        x = self.convT7(x)
        x = F.relu(self.bn7(x), True)
        x = self.convT8(x)
        return torch.tanh(x)


class Discriminator(nn.Module):
    """ Discriminator """
    def __init__(self):
        super(Discriminator, self).__init__()
        #self.ngpu = ngpu

        # input is (channel) x 128 x 128
        self.conv1 = nn.Conv2d(
            cfg.DATASET.CHANNEL, cfg.MODEL.FEATURE_MAPS_D, 4, 2, 1, bias=False
        )
        # state size. (ndf) x 64 x 64
        self.conv2 = nn.Conv2d(
            cfg.MODEL.FEATURE_MAPS_D, cfg.MODEL.FEATURE_MAPS_D*2, 4, 2, 1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_D*2)
        # state size. (ndf*2) x 32 x 32
        self.conv3 = nn.Conv2d(
            cfg.MODEL.FEATURE_MAPS_D*2, cfg.MODEL.FEATURE_MAPS_D*4, 4, 2, 1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_D*4)
        # state size. (ndf*4) x 16 x 16
        self.conv4 = nn.Conv2d(
            cfg.MODEL.FEATURE_MAPS_D*4, cfg.MODEL.FEATURE_MAPS_D*8, 4, 2, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_D*8)
        # state size. (ndf*8) x 8 x 8
        self.conv5 = nn.Conv2d(
            cfg.MODEL.FEATURE_MAPS_D*8, cfg.MODEL.FEATURE_MAPS_D*16, 4, 2, 1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_D*16)
        # state size. (ndf*16) x 4 x 4
        self.conv6 = nn.Conv2d(
            cfg.MODEL.FEATURE_MAPS_D*16, 1, 4, 1, 0, bias=False
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.conv2(x)
        x = F.leaky_relu(self.bn1(x), 0.2, inplace=True)
        x = self.conv3(x)
        x = F.leaky_relu(self.bn2(x), 0.2, inplace=True)
        x = self.conv4(x)
        x = F.leaky_relu(self.bn3(x), 0.2, inplace=True)
        x = self.conv5(x)
        x = F.leaky_relu(self.bn4(x), 0.2, inplace=True)
        return torch.sigmoid(self.conv6(x))


class Discriminator_v2(nn.Module):
    """ Discriminator """
    def __init__(self):
        super(Discriminator_v2, self).__init__()
        #self.ngpu = ngpu


        # input is (channel) x 512 x 512
        self.conv1 = nn.Conv2d(
            cfg.DATASET.CHANNEL, cfg.MODEL.FEATURE_MAPS_D, 4, 2, 1, bias=False
        )
        # input is (ndf) x 256 x 256
        self.conv2 = nn.Conv2d(
            cfg.MODEL.FEATURE_MAPS_D, cfg.MODEL.FEATURE_MAPS_D*2, 4, 2, 1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_D*2)
        # input is (ndf*2) x 128 x 128
        self.conv3 = nn.Conv2d(
            cfg.MODEL.FEATURE_MAPS_D*2, cfg.MODEL.FEATURE_MAPS_D*4, 4, 2, 1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_D*4)
        # state size. (ndf*4) x 64 x 64
        self.conv4 = nn.Conv2d(
            cfg.MODEL.FEATURE_MAPS_D*4, cfg.MODEL.FEATURE_MAPS_D*8, 4, 2, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_D*8)
        # state size. (ndf*8) x 32 x 32
        self.conv5 = nn.Conv2d(
            cfg.MODEL.FEATURE_MAPS_D*8, cfg.MODEL.FEATURE_MAPS_D*16, 4, 2, 1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_D*16)
        # state size. (ndf*16) x 16 x 16
        self.conv6 = nn.Conv2d(
            cfg.MODEL.FEATURE_MAPS_D*16, cfg.MODEL.FEATURE_MAPS_D*32, 4, 2, 1, bias=False
        )
        self.bn5 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_D*32)
        # state size. (ndf*32) x 8 x 8
        self.conv7 = nn.Conv2d(
            cfg.MODEL.FEATURE_MAPS_D*32, cfg.MODEL.FEATURE_MAPS_D*64, 4, 2, 1, bias=False
        )
        self.bn6 = nn.BatchNorm2d(cfg.MODEL.FEATURE_MAPS_D*64)
        # state size. (ndf*64) x 4 x 4
        self.conv8 = nn.Conv2d(
            cfg.MODEL.FEATURE_MAPS_D*64, 1, 4, 1, 0, bias=False
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.conv2(x)
        x = F.leaky_relu(self.bn1(x), 0.2, inplace=True)
        x = self.conv3(x)
        x = F.leaky_relu(self.bn2(x), 0.2, inplace=True)
        x = self.conv4(x)
        x = F.leaky_relu(self.bn3(x), 0.2, inplace=True)
        x = self.conv5(x)
        x = F.leaky_relu(self.bn4(x), 0.2, inplace=True)
        x = self.conv6(x)
        x = F.leaky_relu(self.bn5(x), 0.2, inplace=True)
        x = self.conv7(x)
        x = F.leaky_relu(self.bn6(x), 0.2, inplace=True)
        return torch.sigmoid(self.conv8(x))