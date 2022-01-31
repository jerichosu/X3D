#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 23:09:24 2021

@author: 1517suj
"""


import os
from pathlib import Path
import torch

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.io as io




class VideoDataset(Dataset):

    def __init__(self, directory, mode, n_frames, step_size, transform = None,
                 stats={'mean' : torch.tensor([0.45, 0.45, 0.45]),'std': torch.tensor([0.225, 0.225, 0.225])}):
        
        self.transform = transform
        
    
        folder = Path(directory)/mode  # get the directory of the specified split
        
        self.clip_len = n_frames*step_size # total temporal length (i.e. 16*2 = 32) X3D 

        self.mode = mode

        self.fnames, self.labels = [], []
        
        self.stats = stats
        
        self.step_size = step_size
        
        # for pre-trained weights
        # self.stats_slowfast ={'mean' : torch.tensor([0.45,0.45,0.45]),'std': torch.tensor([0.225,0.225,0.225])}
        
        
        for label in sorted(os.listdir(folder)): # ../RWF-2000/train:
            
            # label: fight/NonfFight
            
            for fname in os.listdir(os.path.join(folder, label)): # ../RWF-2000/train/fight:
                
                # print(fname) # v_ApplyEyeMakeup_g08_c01.avi
                
                self.fnames.append(os.path.join(folder, label, fname))  # ./RWF-2000/train/fight/video001.avi:
                
                self.labels.append(label) # fight/Nonfight
        
        # print(len(self.labels))
        # print((len(self.fnames)))
        
        # # labels: [ApplyEyeMakeup, ..., ..., ...]
        # print(len(labels))
        # # fnames: [the whole address including the label]
        # print(len(self.fnames))

        
        # prepare a mapping between the label names (strings) and indices (ints)
        self.num_class = len(set(self.labels))
        
        # print(self.num_class)
        
        # print(set(self.labels))
        
        # key is class name ---> label:index
        self.class_to_num = dict(zip(set(self.labels), range(self.num_class)))
        
        # print(self.class_to_num)
        
        
        # key is number --> index: label
        self.num_to_class = {v : k for k, v in self.class_to_num.items()}
        # print(self.num_to_class)
        

                
    def __getitem__(self, index):
        
        # print(index)
        
        video_path = self.fnames[index]
        
        # print(len(video_path))
        
        # print(self.labels[index])
    
        label = self.class_to_num[self.labels[index]]
        
        # Load the desired clip
        # video_data, _, _ = io.read_video(filename = video_path)  # [T,H,W,C]
        video_data = io.read_video(filename = video_path, pts_unit='sec')[0]  # [T,H,W,C]

        

        # create an random intial value between the range of (# of frames of the original clip  - temporal window)        
        start_frame = torch.randint(low=0, high=(video_data.size()[0] - self.clip_len), size=(1,))
        video_data = video_data[start_frame:(start_frame + self.clip_len): self.step_size]
        

        # ------------------------------------- COLORJITER FOR TRAINING DATA --------------------------------------
        # REQUIRES [.., C,H,W]
        # if self.mode == 'train':
        #     video_data = video_data.permute(0,3,1,2)  # [T,H,W,C] --> [T,C,H,W]
        #     trans = T.ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1, hue = 0.1)
        #     video_data = trans(video_data)
        #     video_data = video_data.permute(0,2,3,1) # [T,C,H,W] --> [T,H,W,C]
        
        
        # -------------------------------------- TO TENSOR ------------------------------------------------------
        # print(video_data.shape)
        
        
        # Now, it's still [T,H,W,C], so we need re-arrange the tensor dimensions
        video_data = video_data.permute(3,0,1,2) # [T,H,W,C] --> [C,T,H,W] this is what we want
        
        # print(video_data.shape)
        
        #  ------------------------------------ NORMALIZATION -----------------------------------
        # Divide every pixel intensity by 255.0
        video_data = video_data.type(torch.FloatTensor).div_(255.0)
        
        # Pixel intensity from 0 ~ 255 to [-1, +1]
        video_data = video_data.type(torch.FloatTensor).sub_(self.stats['mean'][:,None,None,None]).div_(self.stats['std'][:,None,None,None])
        
        
        # ------------------------  Normalize for slowfast pre-trained weights ------------------- 
        # stats_slowfast ={'mean' : torch.tensor([0.45,0.45,0.45]),'std': torch.tensor([0.225,0.225,0.225])}
        # video_data = video_data.type(torch.FloatTensor).sub_(self.stats_slowfast['mean'][:,None,None,None]).div_(self.stats_slowfast['std'][:,None,None,None])


        # Apply a transform to normalize the video input
        if self.transform is not None:
            # if self.mode == 'train':
            #     video_data = video_data.permute[1,0,2,3] # [C,T,H,W] --> [T,C,H,W]
            #     video_data = self.transform(video_data)
            #     video_data = video_data.permute[1,0,2,3] # [T,C,H,W] --> [C,T,H,W]
            # if self.mode == 'test':
            video_data = self.transform(video_data)
        
        return video_data, label, self.fnames[index]



    def __len__(self):
        return len(self.fnames)


if __name__ == '__main__':
    
    trans=T.Compose([T.RandomResizedCrop(size = 224, scale = (0.8,1.0)),
                     T.RandomHorizontalFlip(),
                     #T.ColorJitter(brightness=0.2, hue=0.3), #[…, 1 or 3, H, W]
                     ])
    

    trans_test=T.Compose([T.Resize(size = [224, 224]),
                             ])
 

    # datapath = '../../UCF-101-split'
    datapath = '../../../SLOWFAST/RWF-2000'
    train_dataloader = \
        DataLoader( VideoDataset(datapath, mode='train', n_frames = 16, step_size = 9, transform = trans), batch_size=4, shuffle=True, num_workers=8)     
    # for step, (buffer, label) in enumerate(train_dataloader):
    #     print("label: ", label)
    #     print(label.shape)
    #     print(label.size(0))
    #     print(buffer.size())  # torch.Size([1, 3, 16, 112, 112])  [batch size, image channels, number of frames, height, width]
    #     break
    
    for step, (videos, labels, _) in enumerate(train_dataloader):  
        print('TRAINING Video batch dimensions:', videos.shape)
        print('TRAINING Video label dimensions:', labels.shape)
        break
