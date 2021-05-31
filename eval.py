#!/usr/local/bin/python
# coding: utf-8
'''
@University  : FEI
@Local       : São Bernardo do Campo, São Paulo, Brazil
@Laboratory  : Image Processing Lab (IPL)
@file        : eval.py
@author      : Lucas Fontes Buzuti
@version     : 1.0.0
@created     : 10/15/2020
@modified    : 10/15/2020
@e-mail      : lucas.buzuti@outlook.com
'''

import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.utils as vutils

from util.config import cfg, cfg_from_file
from util.utils import walk_latent_space, save_model
from util.models.GenerativeAdversarialNets import Generator, Generator_v2



def parse_args():
    parse = argparse.ArgumentParser(description="Train a Paintron Neural Network")
    parse.add_argument(
        '--cfg', dest='cfg_file', 
        help='Path of configure file of the training model',
        default='', type=str
    )
    parse.add_argument(
        '--z_dim', dest='z_dim', 
        help='Number of fake images to be generated',
        default=1, type=int
    )
    parse.add_argument(
        '--ckpt', dest='ckpt', 
        help='Path where the checkpoint are saved',
        default='', type=str
    )
    parse.add_argument(
        '--resolution', dest='resolution', 
        help='Resolution of model',
        default=128, type=int
    )
    args = parse.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    if args.cfg_file == '':
        raise SystemError(
            """ERROR: You need input a path of configure! 
            \npython eval.py --cfg /Path/of/config/file.yaml"""
        )
    else:
        # Load file of config
        cfg_from_file(args.cfg_file)

    # Use CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('device: {}'.format(device))
    # find the best algorithm to use for your hardware
    torch.backends.cudnn.benchmark = True 

    # Open the generator
    if args.resolution == 128:
        netG = Generator()
        checkpoint = torch.load(args.ckpt, map_location=device)
        netG.load_state_dict(checkpoint['netG'])
        netG.to(device)
    elif args.resolution == 512:
        netG = Generator_v2()
        checkpoint = torch.load(args.ckpt, map_location=device)
        netG.load_state_dict(checkpoint['netG'])
        netG.to(device)
    else:
        raise SystemError("ERROR: you need to pick `--resolution` 128 or 512!")

    # Create batch of latent vectors that we will use to 
    # visualize the progression of the generator
    noise = torch.randn(args.z_dim, cfg.MODEL.Z, 1, 1, device=device)
    with torch.no_grad():
        fake = netG(noise).detach()
        
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(
            np.transpose(
                vutils.make_grid(
                    fake.to(device), padding=2, normalize=True, nrow=8
                ).cpu(), (1,2,0)
            )
        )
        plt.savefig(f'./results/img_grid_{args.resolution}px.png')
    

    walk_images = walk_latent_space(device, netG, cfg.MODEL.Z, -2, 2, 16)
    plt.figure(figsize=(16,5))
    plt.axis("off")
    plt.title("Walking Latent Space")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                walk_images.to(device), nrow=16, padding=2, normalize=True
            ).cpu(), (1,2,0)
        )
    )
    plt.savefig(f'./results/walking_{args.resolution}px.png')
