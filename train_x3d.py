#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 16:16:38 2021

@author: 1517suj
"""

# import os
# import argparse
import time

import torch
import torch.nn as nn

import os
from pathlib import Path

from dataset_v2 import VideoDataset

from torch.utils.data import DataLoader

import torchvision.transforms as T
import torchvision.io as io

import torch.nn.functional as F

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter


# import torch.optim as optim
# from torch.optim import lr_scheduler
# from torch.autograd import Variable

# import torchvision
# from torchvision import datasets, transforms
# from torchsummary import summary

# import numpy as np
# from barbar import Bar
# import pkbar
# from apmeter import APMeter

import x3d_net as x3d


# %%

# -----------------------------------------------------------------------------
# SOME HYPER-PARAMETERS

CLIP_LEN = 128
ALPHA = 8
TAU = 64
BATCH_SIZE = 16
datapath = '../SLOWFAST/RWF-2000'

# -----------------------------------------------------------------------------

# DATASET 
trans=T.Compose([T.RandomResizedCrop(size = 224, scale = (0.8,1.0)),
                 T.RandomHorizontalFlip()])


trans_val=T.Compose([T.Resize(size = [224, 224]),
                         ])


# -----------------------------------------------------------------------------
train_loader = \
    DataLoader(VideoDataset(datapath, 
                            mode='train', 
                            clip_len= CLIP_LEN, 
                            alpha = ALPHA, 
                            tau = TAU, 
                            transform = trans_val),
               batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


valid_loader = \
    DataLoader(VideoDataset(datapath, 
                            mode='test', 
                            clip_len= CLIP_LEN, 
                            alpha = ALPHA, 
                            tau = TAU, 
                            transform = trans_val),
               batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    
# -----------------------------------------------------------------------------

# Checking the dataset
for videos, labels in train_loader:  
    print('TRAINING Video batch dimensions:', videos.shape)
    print('TRAINING Video label dimensions:', labels.shape)
    break
 
for videos, labels in valid_loader:  
    print('VALIDATION Video batch dimensions:', videos.shape)
    print('VALIDATION Video label dimensions:', labels.shape)
    break

# %%

# model size
x3d_version = 'M'
# load the model
model = x3d.generate_model(x3d_version=x3d_version, n_classes=400, n_input_channels=3, task='class', dropout=0.5, base_bn_splits=1)

# load pre-trained weights 
load_ckpt = torch.load('models/x3d_multigrid_kinetics_fb_pretrained.pt')
model.load_state_dict(load_ckpt['model_state_dict'])

# change the very last layer, then modify the output
model.fc2 = nn.Linear(in_features = 2048, out_features = 2)


# %% test
# test if it works

# input_tensor = torch.autograd.Variable(torch.rand(32, 3, 16, 224, 224)) # [B (base_bn_splits), C, T, W, H]

# output = model(input_tensor).squeeze(2)
# print(input_tensor.shape)
# print(output.shape)
# # print(output.data)
# print(torch.max(output))
# print(torch.argmax(output))


# %%
# freeze some of layers
model.conv1_s.requires_grad = False
model.conv1_t.requires_grad = False

model.layer1.requires_grad = False
model.layer2.requires_grad = False
# model.layer3.requires_grad = False
# model.layer4.requires_grad = False


# %%

# Hyperparameters
# RANDOM_SEED = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
WEIGHT_DECAY = 0.1
# BATCH_SIZE = params['batch_size']
model_path = './pre_trained_x3d_model.pt'

# Architecture
NUM_CLASSES = 2 # FOR VIOLENCE DETECTION (BIONARY CIASSIFICATION PROBLEM)
DEVICE = 'cuda:0' 
# %%

writer = SummaryWriter()



# %%

# torch.manual_seed(RANDOM_SEED)

##########################
### COST AND OPTIMIZER
##########################

# Load sample videos for model graph

# videos, labels = next(iter(train_loader))
# writer.add_graph(model, videos)

model.to(DEVICE)

# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


print("Preperation Done")
# %%

def compute_accuracy(model, data_loader):
    # model.eval()
    correct_pred, num_examples = 0, 0
    
    for i in tqdm(data_loader):
        features, targets = i
    # for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        # logits, probas = model(features)
        outputs = model(features).squeeze(2)
        
        # print(outputs)
        # print(outputs.data)
        # _, predicted_labels = outputs.topk(1, 1, True, True)
        
        
        _, predicted_labels = torch.max(outputs, 1) # dim = 1 (column), returns the max number of each columns
        
        
        # print(outputs)
        # print(predicted_labels)
        
        # print(predicted_labels.shape)
        # predicted_labels = predicted_labels.t()
        
        
        # _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        
        # correct_pred = predicted_labels.eq(targets.view(1, -1).expand_as(predicted_labels))
        
        correct_pred += (predicted_labels == targets).sum()
        
        # print(correct_pred)
        
    print(num_examples)
    print(correct_pred)
    
    
    return correct_pred.float()/num_examples * 100


def compute_epoch_loss(model, data_loader):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, targets in tqdm(data_loader):
        # for features, targets in data_loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            
            outputs = model(features).squeeze(2)
            
            # logits, probas = model(features)
            # loss = F.cross_entropy(logits, targets, reduction='sum')
            
            loss = F.cross_entropy(outputs, targets, reduction='sum')

            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss
    
    

# %%
    

minibatch_cost, epoch_cost = [], []
all_train_acc, all_valid_acc = [], []

start_time = time.time()

best_acc = 0.0
for epoch in range(NUM_EPOCHS):
    
    # train_acc = []
    # loss = []
    
    model.train()
    # for batch_idx in tqdm(train_loader):
        
    #     print(batch_idx)
        
    #     features, targets = batch_idx
    for batch_idx, (features, targets) in enumerate(tqdm(train_loader)):
        
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
        
        
        # print(targets)
        # print(targets.shape)
        
        ### FORWARD AND BACK PROP
        
        outputs = model(features).squeeze(2)
        
        # print(outputs)
        # print(outputs.data)
        
        cost = F.cross_entropy(outputs, targets)

        
        
        # added for tensorboard
        
        # writer.add_scalar("Loss/train", cost, epoch)
        
        # loss.append(cost.item())
        
        optimizer.zero_grad()
        
        cost.backward()
        
        # minibatch_cost.append(cost)
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        # print(outputs.argmax(dim=-1).shape)
        # print(targets)
        
    #     acc = (outputs.argmax(dim=-1) == targets).float().mean()
    #     train_acc.append(acc)
        
    # train_acc = sum(train_acc) / len(train_acc)
    # train_cost =  sum(loss)/len(loss)
    
    # print(f"[ Train | {epoch + 1:03d}/{NUM_EPOCHS:03d} ] Cost = {train_cost:.5f}, Train_acc = {train_acc:.5f}")
        
        ## LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                    %(epoch+1, NUM_EPOCHS, batch_idx, 
                      len(train_loader), cost))

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        train_acc = compute_accuracy(model, train_loader)
        valid_acc = compute_accuracy(model, valid_loader)

        # train_acc = accuracy(outputs.data, targets, topk=(1, ))
        
        # print(train_acc) 
        
        # valid_acc = accuracy(outputs.data, targets, topk=(1, ))
        
        print('Epoch: %03d/%03d | Train: %.3f%% | Valid: %.3f%%' % (
              epoch+1, NUM_EPOCHS, train_acc, valid_acc))
        

        
        writer.add_scalar("Val_acc",valid_acc,epoch)
        writer.add_scalar("Train_acc",train_acc,epoch)
        
        
        # all_train_acc.append(train_acc)
        # all_valid_acc.append(valid_acc)
        # cost = compute_epoch_loss(model, train_loader)
        # epoch_cost.append(cost)
        
    
            # if the model improves, save a checkpoint at this epoch
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), model_path)
        print('saving model with acc {:.3f}'.format(best_acc))
        
        
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

# Call flush() method to make sure that all pending events have been written to disk.
writer.flush()





