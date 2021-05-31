#!/usr/local/bin/python
# coding: utf-8
'''
@University  : FEI
@Local       : São Bernardo do Campo, São Paulo, Brazil
@Laboratory  : Image Processing Lab (IPL)
@file        : utils.py
@author      : Lucas Fontes Buzuti
@version     : 1.0.0
@created     : 06/01/2020
@modified    : 10/21/2020
@e-mail      : lucas.buzuti@outlook.com
'''

import os
import cv2
import numpy as np
import torch
from datetime import datetime

from .config import cfg
from .RFace.retinaface import RetinaFace



def img_RGB(img):
    """
    img (type -> numpy.array):

    (1) Return: [H, W, C] dim == 3
    (2) Return: [?, H, W, C] dim == 4
    """
    # convert image to [0, 255]
    img = (
        ((img - img.min())*255) / (img.max() - img.min())
    ).astype(np.uint8)
    if len(img.shape) == 3:
        # Change of [C, H, W] to [H, W, C]
        img = np.transpose(img, (1,2,0))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if len(img.shape) == 4:
        # Change of [?, C, H, W] to [?, H, W, C]
        image = np.transpose(img, (0,2,3,1))
        list_batch = [
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image
        ]
        return np.array(list_batch)

def save_log():
    """
    Pointer to generate log file

    Return: Pointer of log file
    """
    if not os.path.exists(cfg.TRAIN.PATH_SAVE_LOG):
        os.mkdir(cfg.TRAIN.PATH_SAVE_LOG)

    log = os.path.join(cfg.TRAIN.PATH_SAVE_LOG, 'TRAINING_LOG_.log')
    if os.path.exists(log):
        os.remove(log)
    
    # Date and time currently
    date = datetime.now()

    fd = open(log, 'w')
    fd.write('@University  : FEI\n')
    fd.write('@Local       : São Bernardo do Campo, São Paulo, Brazil\n')
    fd.write('@Laboratory  : Image Processing Lab (IPL)\n')
    fd.write('@file        : log\n')
    fd.write('@time        : {}\n\n'.format(date.strftime('%d/%m/%Y %H:%M')))
    return fd

def save_model(netG, netD, epoch, file_name):
    """
    Save pytorch model

    netG (type -> torch.nn.Module):
    netD (type -> torch.nn.Module):
    file_name (type -> str):
    """
    if isinstance(netG, torch.nn.DataParallel) and isinstance(netD, torch.nn.DataParallel):
        torch.save(
            {
                'netG': netG.module.state_dict(),
                'netD': netD.module.state_dict(),
                'epoch': epoch
            },
            f'checkpoints/{file_name}'
        )
    else:
        torch.save(
            {
                'netG': netG.state_dict(),
                'netD': netD.state_dict(),
                'epoch': epoch
            }, 
            f'checkpoints/{file_name}'   
        )

def walk_latent_space(device, model, z_dim, start, end, no_of_imgs):
    """
    """
    n = 0
    points = []
    while n < z_dim:
        x = np.linspace(start, end, no_of_imgs)
        x = x[:, np.newaxis]
        points.append(x)
        n += 1

    points = np.hstack(points)
    points = points[:, :, np.newaxis, np.newaxis]
    with torch.no_grad():
        z = torch.tensor(points).to(device, dtype=torch.float)
        gen = model(z)

    return gen


class CroppingFace(object):
    """ Cropping face with RetinaFace """
    def __init__(self):
        self.detector = RetinaFace(
            cfg.RETINA_FACE.MODEL, 0, cfg.RETINA_FACE.GPU_ID, cfg.RETINA_FACE.NETWORK
        )

    def face_detector(self, img, scl):
        count = 1

        im_shape = img.shape
        target_size = scl[0]
        max_size = scl[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        #if im_size_min>target_size or im_size_max>max_size:
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

        scales = [im_scale]
        flip = False

        for c in range(count):
            faces, landmarks = self.detector.detect(
                img, cfg.RETINA_FACE.THRESH, scales=scales, do_flip=flip
            )

        return faces

    def cropped(self, img, faces):
        if faces is not None:
            if faces.shape[0]==1:
                #print('find', faces.shape[0], 'faces')
                for i in range(faces.shape[0]):
                    box = faces[i].astype(np.int)
                    img = img[
                        box[1]-cfg.DATASET.OPEN_BOX:box[3]+cfg.DATASET.OPEN_BOX, 
                        box[0]-cfg.DATASET.OPEN_BOX:box[2]+cfg.DATASET.OPEN_BOX
                    ]
            else:
                img = None
        return img
