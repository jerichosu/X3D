#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 11:26:28 2021

@author: 1517suj
"""

import os
import numpy as np
import cv2
import imageio

import x3d_net as x3d

import torch
import torch.nn as nn
import torchvision.transforms as T


# model size
x3d_version = 'M'
# load the model
model = x3d.generate_model(x3d_version=x3d_version, n_classes=2, n_input_channels=3, task='class', dropout=0, base_bn_splits=1)

# -----------------------------------------------------------------------------
'''
NOTE:
    
if we use DataParallel to train our model, the name of each layer will be added by a
'module.*****', in this case if we load the model directly, the name will not be matched
so we need to do [model = nn.DataParallel(model)], to convert the model name to 
make sure they are the same thing, and now every layer has 'module.****' in front.
'''
# model = nn.DataParallel(model)
# -----------------------------------------------------------------------------


# load pre-trained weights 
model_path = 'pre_trained_x3d_model.pt'
model.load_state_dict(torch.load(model_path))

# load_ckpt = torch.load('X3D/pre_trained_x3d_model.ckpt')
# model.load_state_dict(load_ckpt['model_state_dict'])

# change the very last layer, then modify the output
# model.fc2 = nn.Linear(in_features = 2048, out_features = 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Device being used:", device)

model.to(device)
model.eval()
# %% test

# input_tensor = torch.autograd.Variable(torch.rand(1, 3, 16, 224, 224))
# input_tensor = input_tensor.to(device)

# output = model(input_tensor).squeeze(2)

# print(output)
# print(output.argmax)


# !!!!!!!!!!!!!!!!
# data normalization!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!
stats={'mean' : torch.tensor([0.45, 0.45, 0.45]),'std': torch.tensor([0.225, 0.225, 0.225])}

# %%

trans_val=T.Compose([T.Resize(size = [224, 224]),])

# create a simple mapping between predicted label and its corresponding results
labels = {0: 'Fight', 1: 'NonFight'}


# dir = '../../../../DATASETS/Surv/fight'
dir = '../../../../DATASETS/CCTV_Fights/MPEG_Compressed/mpeg-001-100'
# dir = '../../../../DATASETS/CCTV_Fights/MPEG_Compressed/mpeg-301-400' # 1/2/3 8 bad/6/7/12
# dir = '../../../../SLOWFAST/RWF-2000/test/Fight'


# dir = 'SLOWFAST/RWF-2000/test/NonFight'
# dir = 'SLOWFAST/RWF-2000/test/Fight'
# dir = 'SLOWFAST/RWF-2000/train/Fight'
clips = os.listdir(dir)

path = os.path.join(dir,os.listdir(dir)[25])


# path = 'SLOWFAST/RWF-2000/test/NonFight/2lrARl7utL4_2.avi'

cap = cv2.VideoCapture(path)

retaining = True
kk = 0
image_list = []

clip = []
while retaining:
    retaining, frame = cap.read()
    
    if not retaining and frame is None:
        continue
    # tmp_ = center_crop(cv2.resize(frame, (171, 128)))
    # tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
    
    # i = 0
    # i += 1
    # if i == 0 and i % 7 == 0:
    #     clip.append(frame)
        
    clip.append(frame)
    if len(clip) == 16:
        inputs = np.array(clip).astype(np.float32)
        
        # inputs = np.expand_dims(inputs, axis=0)
        # inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
        
        # convert from numpy array to tensor
        inputs = torch.from_numpy(inputs) # torch.Size([16, 360, 640, 3])
        inputs = inputs.permute(3,0,1,2) # torch.Size([3, 16, 360, 640])
        # print(inputs.shape)
        
        # change the height and width to fit the model
        inputs = trans_val(inputs) # torch.Size([3, 16, 224, 224])
        
        # increase the dimension
        inputs = torch.unsqueeze(inputs, 0) # torch.Size([1, 3, 16, 224, 224])
        
        
        # Data normalization
        # Divide every pixel intensity by 255.0
        inputs = inputs.type(torch.FloatTensor).div_(255.0)
        # Pixel intensity from 0 ~ 255 to [-1, +1]
        inputs = inputs.type(torch.FloatTensor).sub_(stats['mean'][:,None,None,None]).div_(stats['std'][:,None,None,None])
        
        
        inputs = inputs.to(device)
        # inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
        with torch.no_grad():
            outputs = model(inputs).squeeze(2)
            # outputs = model.forward(inputs).squeeze(2)
            
        # compute the probability    
        m = nn.Softmax(dim=1)
        # probs = torch.max(m(outputs))
        probs_fight = m(outputs)[0][0]
        
        # check!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # probs = torch.max(outputs)
        # print(probs_fight)
        predicted_labels = torch.argmax(outputs)
        # _, predicted_labels = torch.max(outputs, 1)
        # print(int(predicted_labels))
        
        # print(outputs.shape)
        
        # label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

        cv2.putText(frame, labels[int(predicted_labels)], (5, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 0, 255), 1)
        
        cv2.putText(frame, "Prob_fight: %.4f" % probs_fight, (5, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255), 1)
        # print(len(clip))
        clip.pop(0)
        # print(len(clip))
        # break
    
    cv2.imshow('result', frame)
    
    # make sure don't make gif too big
    kk += 1
    # print(kk)
    if kk > 280 and kk < 600:
        print(kk)
        image_list.append(frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    # cv2.waitKey(30)

cap.release()
cv2.destroyAllWindows()

# # Convert to gif using the imageio.mimsave method
imageio.mimsave('./video.gif', image_list, fps=25)
    
    








