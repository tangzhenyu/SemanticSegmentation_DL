#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Created on 2017/08/23
@author: renjiao
'''
import os
from scipy.io import loadmat as sio
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

path='/home/dl/DL_dataset/VOCdevkit/trainval'
files=os.listdir(path)
labels_path=os.path.join(path,'labels')

for afile in files:
    file_path=os.path.join(path,afile)
    if os.path.isfile(file_path):
        if os.path.getsize(file_path)==0:
            continue
        mat_idx=afile[:afile.find('.mat')]
        mat_file=sio(file_path)
        mat_file=np.array(mat_file['LabelMap'])
       # print mat_file.keys()
        mat_file=mat_file.astype(np.uint8)
        label_img=Image.fromarray(mat_file.reshape(mat_file.shape[0],mat_file.shape[1]))
        dst_path=os.path.join(labels_path,mat_idx+'.png')
        label_img.save(dst_path)

