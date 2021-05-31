#!/usr/local/bin/python
# coding: utf-8
'''
@University  : FEI
@Local       : São Bernardo do Campo, São Paulo, Brazil
@Laboratory  : Image Processing Lab (IPL)
@file        : config.py
@author      : Lucas Fontes Buzuti
@version     : 1.0.0
@created     : 14/10/2020
@modified    : 14/10/2020
@e-mail      : lucas.buzuti@outlook.com
'''

import os
import numpy as np
from easydict import EasyDict as edict



__C = edict()
cfg = __C


__C.CONFIG_NAME = ''


# Dataset parameters
__C.DATASET = edict()
__C.DATASET.NAME = 'UNIFESP'
__C.DATASET.DIR = ''
__C.DATASET.HEIGHT_PXL = 233
__C.DATASET.WIDTH_PXL = 450
__C.DATASET.CHANNEL = 3
__C.DATASET.OPEN_BOX = 0
__C.DATASET.WORKERS = 0


# RetinaFace parameters
__C.RETINA_FACE = edict()
__C.RETINA_FACE.MODEL = ''
__C.RETINA_FACE.GPU_ID = 0
__C.RETINA_FACE.NETWORK = 'net3'
__C.RETINA_FACE.THRESH = .8


# Model parameters
__C.MODEL = edict()
__C.MODEL.Z = 0
__C.MODEL.FEATURE_MAPS_G = 0
__C.MODEL.FEATURE_MAPS_D = 0


# Training parameters
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 16
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.NUMBER_GPU = 0

__C.TRAIN.LEARNING_RATE = 1e-5
__C.TRAIN.BETA1 = 1e-5
__C.TRAIN.SMOOTHING = .9

# Establish convention for real and fake labels during traing
__C.TRAIN.REAL_LABEL = 1. # Not change
__C.TRAIN.FAKE_LABEL = 0. # Not change

__C.TRAIN.FREQUENCY_TENSORB = 60


# Plot parameters



def _merge(a, b):
    """
    Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """ Load a config file and merge it into the default options. """
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge(yaml_cfg, __C)
