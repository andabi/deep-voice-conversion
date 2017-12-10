# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import glob
import numpy as np


def split_path(path):
    '''
    'a/b/c.wav' => ('a/b', 'c', 'wav')
    :param path: filepath = 'a/b/c.wav'
    :return: basename, filename, and extension = ('a/b', 'c', 'wav')
    '''
    basepath, filename = os.path.split(path)
    filename, extension = os.path.splitext(filename)
    return basepath, filename, extension


def remove_all_files(prefix):
    files = glob.glob(prefix + '*')
    for f in files:
        os.remove(f)


def normalize_0_1(values, max, min):
    normalized = np.clip((values - min) / (max - min), 0, 1)
    return normalized


def denormalize_0_1(normalized, max, min):
    values =  np.clip(normalized, 0, 1) * (max - min) + min
    return values
