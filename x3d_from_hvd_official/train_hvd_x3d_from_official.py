#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 14:21:38 2021

@author: 1517suj
"""


import argparse
import os
from distutils.version import LooseVersion
from filelock import FileLock

import torch.multiprocessing as mp
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
import torch.utils.data.distributed
import horovod.torch as hvd
import torchvision.transforms as T
from torch.utils.data import DataLoader


from dataset_v2_wfname import VideoDataset
from x3d import generate_model

# %%

# input_tensor = torch.autograd.Variable(torch.rand(1, 3, 16, 224, 224)) # [B, C, T, H, W]
    
# x3d_version = 'M'
# model = generate_model(x3d_version, n_classes = 400)
# output = model(input_tensor)
# print(input_tensor.shape)
# print(output.shape)

# print(torch.max(output))
# print(torch.argmax(output))

# model_path = 'X3D-M-RENAME.pt'
# model.load_state_dict(torch.load(model_path))
    
    


# %%
def train_mixed_precision(model, train_sampler, train_loader, optimizer, epoch, scaler):
    # set the model to be in training mode
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target, _) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(data)
            criterion = nn.CrossEntropyLoss().cuda()
            loss = criterion(output, target)

        scaler.scale(loss).backward() # ----------
        # Make sure all async allreduces are done
        optimizer.synchronize() # -------------
        # In-place unscaling of all gradients before weights update
        scaler.unscale_(optimizer) # --------------
        with optimizer.skip_synchronize():
            scaler.step(optimizer)
        # Update scaler in case of overflow/underflow
        scaler.update() # ------------------

        if batch_idx % args.log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss Scale: {}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                100. * batch_idx / len(train_loader), loss.item(), scaler.get_scale()))
            

# %%
def train(model, train_sampler, train_loader, optimizer, epoch):
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target, _) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss().cuda()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                100. * batch_idx / len(train_loader), loss.item()))


# wrap up training/bvalidation results
def metric_average(val, name):
    # print(val)
    tensor = val.clone().detach()
    # tensor = torch.tensor(val)
    # avg_tensor = hvd.allreduce(tensor, name=name, op=hvd.Sum)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()



# # wrap up training/bvalidation results
# def metric_sum(val, name):
#     tensor = val.clone().detach()
#     sum_tensor = hvd.allreduce(tensor, name=name, op=hvd.Sum)
#     return sum_tensor.item()


def test(model, test_loader, test_sampler):
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    with torch.no_grad():
        for data, target, _ in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            # criterion = nn.CrossEntropyLoss(reduction='sum').cuda()
            criterion = nn.CrossEntropyLoss().cuda()
            test_loss += criterion(output, target)
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()
    
        # Horovod: use test_sampler to determine the number of examples in
        # this worker's partition.
        test_loss /= len(test_sampler)
        test_accuracy /= len(test_sampler)
    
        # Horovod: average metric values across workers.
        test_loss = metric_average(test_loss, 'avg_loss')
        test_accuracy = metric_average(test_accuracy, 'avg_accuracy')
    
        # Horovod: print output only on first rank.
        if hvd.rank() == 0:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
                test_loss, 100. * test_accuracy))
        
        
        
# %%     
# def correct(output, target, topk=(1,), fnames = None):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)

#     # For mini-batches with multiple output predictions
#     if len(list(output.size())) > 1:
#         # Return the indices of where the maxk largest probabilities are (sorted
#         # with the highest probabilities first), then transpose
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
        
#         # Return a boolean tensor describing element-wise equality 
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#     # For mini-batches that only have one output
#     else:
#         pred = torch.argmax(output)
        
#         # Return a boolean tensor describing element-wise equality
#         correct = pred.eq(target)
    
#     # print(correct)
#     # correct: tensor([[ True, False, False,  True, False, False,  True, False, False, False,
#     # False, False, False,  True,  True,  True]], device='cuda:0')
#     # ???
    
#     # # Report misclassifications if the filenames were passed (validation in
#     # # the last epoch)
#     if fnames is not None:
        
#         # Print out the file name of each misclassified clip
#         if True:
#             if not correct:
#                 # Split the path using the OS path separator, only use the last
#                 # two divisions for simplicity
#                 segs = fnames.split(os.path.sep)
#                 print(os.path.join(segs[-2],segs[-1]) + ' was misclassified.')
                
#     res = []
#     for k in topk:
#         # Added to make correct contiguous in memory (view() only works on
#         # contiguous arrays)
#         correct = correct.contiguous()
#         # Slice the first k elements of correct and add them up.
#         correct_k = correct[:k].view(-1).float().sum(0)
#         # Store in list to return
#         res.append(correct_k)
#     return res
        


# # Begin test function --------------------------------------------------------
# def validation(model, val_dataloader, val_sampler, epoch, criterion, hightest_val_acc):

#     # Set model to eval mode
#     model.eval()
    
#     top1 = 0.0
    
#     # Stop updating gradients, since this is testing
#     with torch.no_grad():
        
#         for batch_idx, (inputs, labels, fnames) in enumerate(val_dataloader):
            
#             # Move inputs/labels to GPU
#             inputs = inputs.cuda()
#             labels = labels.cuda()
            
#             # Get the original outputs of the model
#             outputs = model(inputs)
            
#             # Only pass the filenames (for explaining misclassification) on the
#             # final epoch
#             # if epoch < NUM_EPOCH-1:
#             fnames = None
            
#             corr1 = correct(outputs.data, labels, (1,), fnames)[0]
            
#             # Sum correct predictions from this worker across batches
#             top1 += corr1
                
#         # After all batches for this worker, calculate total accuracy --------
    
#         # Sum metric values across workers
#         # top1 = metric_sum(top1, 'sum_top1')
#         top1 = metric_average(top1, 'avg_accuracy')
            
#         # Convert to per-sample values
#         top1 = (top1*100)/(len(val_sampler) * hvd.size())
        
#         if top1 > hightest_val_acc:
#             hightest_val_acc = top1
    
#     return top1, hightest_val_acc
        
# %%
def stage_freeze(which_stage, model): #['none', 'up2conv1', 'up2res2', 'up2res3', 'up2res4', 'up2res5']
    
    stage_options = ['none', 'up2conv1', 'up2res2', 'up2res3', 'up2res4', 'up2res5']
    stage = stage_options[which_stage]
    
    if stage == 'none':  # train everything
        return model
    
    if stage == 'up2conv1': # train layer1 + layer2 + layer3 + layer4 + fc layers
        model.conv1_s.requires_grad = False
        model.conv1_t.requires_grad = False
        model.bn1.requires_grad = False
        
    if stage == 'up2res2': # train layer2 + layer3 + layer4 + fc layers
        model.conv1_s.requires_grad = False
        model.conv1_t.requires_grad = False
        model.bn1.requires_grad = False
        
        model.layer1.requires_grad = False
        
    if stage == 'up2res3': # train layer3 + layer4 + fc layers
        model.conv1_s.requires_grad = False
        model.conv1_t.requires_grad = False
        model.bn1.requires_grad = False
        
        model.layer1.requires_grad = False
        model.layer2.requires_grad = False
        
    if stage == 'up2res4': # train layer4 + fc layers
        model.conv1_s.requires_grad = False
        model.conv1_t.requires_grad = False
        model.bn1.requires_grad = False
        
        model.layer1.requires_grad = False
        model.layer2.requires_grad = False
        model.layer3.requires_grad = False
        
    if stage == 'up2res5': # only train fc layers
        model.conv1_s.requires_grad = False
        model.conv1_t.requires_grad = False
        model.bn1.requires_grad = False
        
        model.layer1.requires_grad = False
        model.layer2.requires_grad = False
        model.layer3.requires_grad = False
        model.layer4.requires_grad = False
    return model

# %%
def main(args):
    # Horovod: initialize library.
    hvd.init()
    # torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        # torch.cuda.manual_seed(args.seed)
    else:
        if args.use_mixed_precision:
            raise ValueError("Mixed precision is only supported with cuda enabled.")

    if (args.use_mixed_precision and LooseVersion(torch.__version__)
            < LooseVersion('1.6.0')):
        raise ValueError("""Mixed precision is using torch.cuda.amp.autocast(),
                            which requires torch >= 1.6.0""")
                            
    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(torch.get_num_threads())

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    # %%-------------------------------------------------------------------------
    # DATASET
    
    # define image augmentation for training and testing datasets
    trans=T.Compose([T.RandomResizedCrop(size = args.image_size, scale = (0.8,1.0)),
                 T.RandomHorizontalFlip()])
    
    # for testing datset, do not use data augmentation
    trans_test=T.Compose([T.Resize(size = [args.image_size, args.image_size]),
                                     ])
    
    
    # 
    data_dir = args.data_dir # path to the dataset
    with FileLock(os.path.expanduser("~/.horovod_lock")): #expand ~/.horovod_lock  -->  /home/MARQNET/1517suj/.horovod_lock
        # training dataset
        train_dataset = \
                VideoDataset(data_dir, mode='train', n_frames = 16, step_size = args.step_size,  transform = trans)
    
    
    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

        

    # validation dataset            
    test_dataset = VideoDataset(data_dir, mode='test', n_frames = 16, step_size = args.step_size, transform = trans_test)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank()) #-----------------split the data on separate gpus   
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              sampler=test_sampler, **kwargs)
                
            
        # Checking the dataset
    for videos, labels, _ in train_loader:
        if hvd.rank() == 0:
            print('TRAINING Video batch dimensions:', videos.shape)
            print('TRAINING Video label dimensions:', labels.shape)
        break
     
    for videos, labels, _ in test_loader:  
        if hvd.rank() == 0:
            print('VALIDATION Video batch dimensions:', videos.shape)
            print('VALIDATION Video label dimensions:', labels.shape)
        break
    
    
    if (hvd.rank() == 0):
        print("Number of training clips: " + str(len(train_dataset)))
        print("Number of validation clips: " + str(len(test_dataset)))
            
    #%% -------------------------------------------------------------------------
    # MODEL
    
    # define X3D model
    x3d_version = args.model_size # size M default
    model = generate_model(x3d_version, n_classes = 400)
    # load pretrained weights
    model_path = args.model_path # default: 'X3D-M-RENAME.pt'
    model.load_state_dict(torch.load(model_path))
    # modify the last layer to match the need:
    # change the very last layer, then modify the output
    model.fc2 = nn.Linear(in_features = 2048, out_features = 2)
    # decides how many layers needs to be frozen
    model = stage_freeze(args.freeze_stage, model)
    
    
    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()
            # print(111)
    
    # -------------------------------------------------------------------------
    # Horovod: scale learning rate by lr_scaler.
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr * lr_scaler,
                              momentum=args.momentum)
    if args.optimizer == 'adam':
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        optimizer = optim.Adam(model.parameters(), lr=args.lr * lr_scaler,)
        
    print('optimizer used:' + args.optimizer)


    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Adasum if args.use_adasum else hvd.Average,
                                         gradient_predivide_factor=args.gradient_predivide_factor)


    if args.use_mixed_precision:
        # Initialize scaler in global scale
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, args.epochs + 1):
        if args.use_mixed_precision:
            train_mixed_precision(model, train_sampler, train_loader, optimizer, epoch, scaler)
        else:
            train(model, train_sampler, train_loader, optimizer, epoch)
        print('begin test')
        # Keep test in full precision since computation is relatively light.
        test(model, test_loader, test_sampler)
        
        

# %%    

if __name__ == '__main__':
    
    # Training settings
    parser = argparse.ArgumentParser(description='Training X3D, modified from official PyTorch MNIST Example')
    
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 16)')
    
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    
    parser.add_argument('--optimizer', default='adam', 
                        help='what optimizer to use')
    
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.1)')
    
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    
    # parser.add_argument('--seed', type=int, default=42, metavar='S',
    #                     help='random seed (default: 42)')
    
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--fp16-allreduce', action='store_true', default=True,
                        help='use fp16 compression during allreduce')
    
    parser.add_argument('--use-mixed-precision', action='store_true', default = True,
                        help='use mixed precision for training')
    
    parser.add_argument('--use-adasum', action='store_true', default=False,
                        help='use adasum algorithm to do reduction')
    
    parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                        help='apply gradient predivide factor in optimizer (default: 1.0)')
    
    parser.add_argument('--data-dir', default='../../../SLOWFAST/RWF-2000',
                        help='location of the training dataset in the local filesystem (will be downloaded if needed)')
    
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers, default = 4')
    
    parser.add_argument('--model_size', type=str, default='M',
                        help='size of the x3d model, default = size M')
    
    parser.add_argument('--model_path', type=str, default='X3D-M-RENAME.pt',
                        help='pre-trained weights of the model, default = X3D-M-RENAME.pt')
                        
    parser.add_argument('--freeze_stage', type=int, default=3, # ['none', 'up2conv1', 'up2res2', 'up2res3', 'up2res4', 'up2res5']
                        help='which stages need to be frozen when using transfer learning, ranging from 0~5, default = 3 (allow layer 4,5 and fc layer to update)')
    
    parser.add_argument('--step_size', type=int, default=9, 
                        help='step size between each frame, like the sampling rate')
    
    parser.add_argument('--image_size', type=int, default=256, 
                        help='image size')
    
    # parser.add_argument('--freeze_stage', type=int, default=3, # ['none', 'up2conv1', 'up2res2', 'up2res3', 'up2res4', 'up2res5']
    #                     help='which stages need to be frozen when using transfer learning, ranging from 0~5, default = 3 (allow layer 4,5 and fc layer to update)')
    
    
                        
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    
    # run model
    main(args)

