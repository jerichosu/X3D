#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 16:16:43 2021

@author: 1517suj
"""

import os
import numpy as np
import cv2
import x3d_net as x3d

import torch
import torch.nn as nn
import torchvision.transforms as T

# %%
# model size
x3d_version = 'M'
# load the model
model = x3d.generate_model(x3d_version=x3d_version, n_classes=2, n_input_channels=3, task='class', dropout=0, base_bn_splits=1)

'''
NOTE:
    
if we use DataParallel to train our model, the name of each layer will be added a
'module.*****', in this case if we load the model directly, the name will not be matched
so we need to do [model = nn.DataParallel(model)], to convert the model name to 
be the same, and now every model has 'module.****' in front.
'''
model = nn.DataParallel(model)
# for name, layer in model.named_modules():
#     print(name, layer)
# %%
# load pre-trained weights 
model_path = 'pre_trained_x3d_model.pt'
# mt = torch.load(model_path)
# for key, value in mt.items():
#     print(key)
model.load_state_dict(torch.load(model_path))


# %%
# load_ckpt = torch.load('X3D/pre_trained_x3d_model.ckpt')
# model.load_state_dict(load_ckpt['model_state_dict'])

# change the very last layer, then modify the output
# model.fc2 = nn.Linear(in_features = 2048, out_features = 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Device being used:", device)