#!/usr/local/bin/python
# coding: utf-8
'''
@University  : FEI
@Local       : São Bernardo do Campo, São Paulo, Brazil
@Laboratory  : Image Processing Lab (IPL)
@file        : database.py
@author      : Lucas Fontes Buzuti
@version     : 1.0.0
@created     : 06/01/2020
@modified    : 14/10/2020
@e-mail      : lucas.buzuti@outlook.com
'''

import os
import shutil

import cv2
import numpy as np
from PIL import Image
from torch.utils import data

from .config import cfg
from .utils import CroppingFace



class Dataset(data.Dataset):
    """ Dataset """
    def __init__(self, list_path, retinaface=False, transform=None):
        self.data = list_path
        self.retinaface = retinaface
        self.transform = transform

        if self.retinaface:
            print("Enabled RetinaFace.")
            self.RF = CroppingFace()
            self.scale = [cfg.DATASET.HEIGHT_PXL+100, cfg.DATASET.WIDTH_PXL+100]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx])
        if self.retinaface:
            bbox_face = self.RF.face_detector(img, self.scale)
            img = self.RF.cropped(img, bbox_face)

        if self.transform is not None:
            img = Image.fromarray(img)
            b, g, r = img.split()
            img = Image.merge('RGB', (r, g, b))
            img = self.transform(img)

        return img


class DatasetTensor(data.Dataset):
    """ Dataset """
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx]
    