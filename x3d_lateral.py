#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:03:21 2021

@author: 1517suj
"""

import torch
import torch.nn as nn


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=(1,stride,stride),
                     padding=1,
                     bias=False,
                     groups=in_planes
                     )


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=(1,stride,stride),
                     bias=False)


class SqEx(nn.Module):
    # Squeeze-and-Excitation.  This implements an SE unit as described in
    # Section 3 of the original Squeeze-and-Excitation paper.
    def __init__(self,in_plane):
        super(SqEx, self).__init__()
        
        # Define layers for Squeeze-&-Excitation units
        self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))
        # Note that here, fc layers are imitated by using equivalent 1x1x1
        # 3D convolutions
        self.fc_1 = nn.Conv3d(in_plane, self.round_width(in_plane), kernel_size=1, stride=1)
        self.relu_se = nn.ReLU(inplace=True)
        self.fc_2 = nn.Conv3d(self.round_width(in_plane), in_plane, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        
    
    def round_width(self, width, multiplier=0.0625, min_width=8, divisor=8):
        """
        Round width of filters based on width multiplier
        Args:
            width (int): the channel dimensions of the input.
            multiplier (float): the multiplication factor.
            min_width (int): the minimum width after multiplication.
            divisor (int): the new width should be dividable by divisor.
        """
        if not multiplier:
            return width

        width *= multiplier
        min_width = min_width or divisor
        width_out = max(
            min_width, int(width + divisor / 2) // divisor * divisor
        )
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)
        
    
    def forward(self,x):
        # Pool all inputs to be 1x1x1 ("Squeeze" part of the SE unit). We
        # now have a 1x1x1 image for each channel.  This is Equation 2 of 
        # the original Squeeze-and-Excitation paper.
        se_w = self.global_pool(x)
        # The next 4 lines produce Equation 3 of the original Squeeze-and-
        # Excitation paper
        se_w = self.fc_1(se_w)
        se_w = self.relu_se(se_w)
        se_w = self.fc_2(se_w)
        se_w = self.sigmoid(se_w)
        # The next line produces Equation 4 of the original Squeeze-and-
        # Excitation paper
        x = x * se_w
        
        return x
    
    
    


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None, index=0):
        super().__init__()

        self.index = index # used for squeeze-and-excitation
        
        self.conv1 = conv1x1x1(in_planes, planes[0])
        # ------------------------------------------------------
        # pre-trained weights might need that costumized function 
        self.bn1 = nn.BatchNorm3d(planes[0], affine=True, track_running_stats=True)
        # ------------------------------------------------------
        self.conv2 = conv3x3x3(planes[0], planes[0], stride)
        # ------------------------------------------------------
        self.bn2 = nn.BatchNorm3d(planes[0], affine=True, track_running_stats=True)
        # ------------------------------------------------------
        self.conv3 = conv1x1x1(planes[0], planes[1])
        # ------------------------------------------------------
        self.bn3 = nn.BatchNorm3d(planes[1], affine=True, track_running_stats=True)
        # ------------------------------------------------------
        self.swish = nn.SiLU()
        self.relu = nn.ReLU(inplace=True)
        
        # ------------------- squeeze and excitation --------------------------
        if self.index % 2 == 0:
            self.SE = SqEx(planes[0])
            
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # Squeeze-and-Excitation
        if self.index % 2 == 0:
            out = self.SE(out)
            
            
        out = self.swish(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            
        # if not like res2->res3, residual = x!, else residual need conv1x1x1 w/ stride 2!!!!
        out += residual
        out = self.relu(out)

        return out


# %%----------------------------------------------------------------------------------------
class ResNet(nn.Module):

    def __init__(self,
                 block, # Bottleneck
                 layers, # get_blocks(x3d_version) -> size M: [3,5,11,7], 3 blocks, 5 blocks...so forth
                 layers_flow, # number of each flow block repeated
                 block_inplanes, # get_inplanes(x3d_version) -> size M: [(54, 24), (108, 48), (216, 96), (432, 192)] # of channels @ each layer
                 block_inflows, # size M: [(6, 3), (13, 6), (27, 12), (54, 24)] # of output channels @ each FLOW layer
                 n_input_channels=3,
                 flow_in_channels = 2,
                 dropout=0.5,
                 n_classes=400):
        super().__init__() # inheret functions from nn.Module.__init__()


        # print(block_inplanes) # size M: [(54, 24), (108, 48), (216, 96), (432, 192)]
        
        self.index = 0 # used for squeeze and excitation in BottleNeck


        self.in_planes = block_inplanes[0][1] #24 for M, # of out channels for conv1 x3d layer
        self.in_planes_flow = block_inflows[0][1] # 3 for M, # of out channels for conv1 flow layer
        
        
        # -> 16x224x224
        
        
        self.conv1_s = nn.Conv3d(n_input_channels, # 3
                               self.in_planes, # 24
                               kernel_size=(1, 3, 3),
                               stride=(1, 2, 2),
                               padding=(0, 1, 1),
                               bias=False)
        
        # -> 16x112x112
        
        # temporal conv
        self.conv1_t = nn.Conv3d(self.in_planes, #24
                               self.in_planes, #24
                               kernel_size=(5, 1, 1), # in paper it's 3, but 5 makes sense if calculate
                               # kernel_size=(3, 1, 1),
                               stride=(1, 1, 1),
                               padding=(2, 0, 0), #padding=(1, 0, 0) if kernel size is (3,1,1)
                               bias=False,
                               groups=self.in_planes) # depthwise convolution
        # -> 16x112x112
        
        
        self.bn1 = nn.BatchNorm3d(self.in_planes, affine=True, track_running_stats=True)
        
        self.relu = nn.ReLU(inplace=True)
        # -> 16x112x112
        
        
        # ------------------ res2 block ---------------------------------------
        self.layer1 = self._make_layer(block, # Bottleneck
                                       block_inplanes[0], # (54,24)
                                       layers[0], #3
                                       stride=2,
                                       in_planes_slow = block_inplanes[0][1],# + block_inflows[0][1], # 24 + 3, 3 is from flow channel
                                       )
        # -> 16x56x56
        
        
        # ------------------ res3 block ---------------------------------------
        self.layer2 = self._make_layer(block, # Bottleneck
                                       block_inplanes[1], # (108,48)
                                       layers[1], #5
                                       stride=2,
                                       in_planes_slow = block_inplanes[0][1],# + block_inflows[0][1], # 24 + 3, 3 is from flow channel
                                       )
        # -> 16x28x28
        
        
        # ------------------ res4 block ---------------------------------------
        self.layer3 = self._make_layer(block, # Bottleneck
                                       block_inplanes[2], # (216,96)
                                       layers[2], # 11
                                       stride=2,
                                       in_planes_slow = block_inplanes[1][1] + block_inflows[1][1],
                                       ) # 48 + 6, 6 is from flow channel
        # -> 16x14x14
        
        
        # ------------------ res5 block ---------------------------------------
        self.layer4 = self._make_layer(block, # Bottleneck
                                       block_inplanes[3], # (432,192)
                                       layers[3], # 7
                                       stride=2,
                                       in_planes_slow = block_inplanes[2][1], #+ block_inflows[2][1],
                                       ) # 96 + 12, 12 is from flow channel
        # -> 16x7x7
        
        
        # ------------------ conv5 --------------------------------------------
        self.conv5 = nn.Conv3d(block_inplanes[3][1], #+ block_inflows[3][1], #  192, + 24 = 216, 24 is from flow channel
                               block_inplanes[3][0],
                               kernel_size=(1, 1, 1),
                               stride=(1, 1, 1),
                               padding=(0, 0, 0),
                               bias=False)
        self.bn5 = nn.BatchNorm3d(block_inplanes[3][0], affine=True, track_running_stats=True)
        
        # -> 16x7x7
        
        
        # ------------------ pool5 --------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # -> 16x7x7
        
        
        # ------------------ fc1 ----------------------------------------------
        self.fc1 = nn.Conv3d(block_inplanes[3][0], 2048, bias=False, kernel_size=1, stride=1)
        # -> 1x1^2, 2048
        
        
        # ------------------ fc2 ----------------------------------------------
        self.fc2 = nn.Linear(2048, n_classes)
        # -> 2048, # of classes
        
        
        self.dropout = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # for initialziation
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight,
        #                                 mode='fan_out',
        #                                 nonlinearity='relu')
        # ---------------------------------------------------------------------
        
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # ---------------------------------------------------------------------
        # FLOW STREAM:%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # ---------------------------------------------------------------------
        # conv1 stage:
        self.conv1_s_flow = nn.Conv3d(flow_in_channels, # 2
                                   self.in_planes_flow, # 3
                                   kernel_size=(1, 3, 3),
                                   stride=(1, 2, 2),
                                   padding=(0, 1, 1),
                                   bias=False)
        
        # -> 16x112x112
        
        # temporal conv
        self.conv1_t_flow = nn.Conv3d(self.in_planes_flow, #3
                               self.in_planes_flow, #3
                               kernel_size=(5, 1, 1), # in paper it's 3, but 5 makes sense if calculate
                               # kernel_size=(3, 1, 1),
                               stride=(1, 1, 1),
                               padding=(2, 0, 0), #padding=(1, 0, 0) if kernel size is (3,1,1)
                               bias=False,
                               groups=self.in_planes_flow) # depthwise convolution
        # -> 16x112x112
        
        
        self.bn1_flow = nn.BatchNorm3d(self.in_planes_flow, affine=True, track_running_stats=True)
        
        self.relu_flow = nn.ReLU(inplace=True)
        # -> 16x112x112
        # ---------------------------------------------------------------------
        
        # ------------------ res2 block ---------------------------------------
        self.layer1_flow = self._make_layer_flow(block, # Bottleneck
                                        block_inflows[0], # (6,3) # of channels for 1&2 layers in the block: 6, 3rd layers: 3 (# of output channels)
                                        layers_flow[0], #1, how many time each block is repeated
                                        stride=2)
        # # -> 16x56x56
        
        # ------------------ res3 block ---------------------------------------
        self.layer2_flow = self._make_layer_flow(block, # Bottleneck
                                        block_inflows[1], # (108//8 = 13, 48//8 = 6)
                                        layers_flow[1], #
                                        stride=2)
        # # -> 16x28x28
        
        
        # # ------------------ res4 block ---------------------------------------
        # self.layer3_flow = self._make_layer_flow(block, # Bottleneck
        #                                block_inflows[2], # (216//8 = 27, 96//8 = 11)
        #                                layers_flow[2], # 11
        #                                stride=2)
        # # -> 16x14x14
        
        
        # # ------------------ res5 block ---------------------------------------
        # self.layer4_flow = self._make_layer_flow(block, # Bottleneck
        #                                block_inflows[3], # (432,192)
        #                                layers_flow[3], # 7
        #                                stride=2)
        # # -> 16x7x7
        
        # # ---------------------------------------------------------------------
    def FlowPath(self, input):
        lateral = []   # input: 16x224x224
        # print(input.shape)
        # stage 1
        x = self.conv1_s_flow(input)
        x = self.conv1_t_flow(x)
        x = self.bn1_flow(x)
        x = self.relu_flow(x) # 16x112x112
        # print(x.shape)
        lateral.append(x)
        # stage 2
        x = self.layer1_flow(x) # 16x56x56
        # print(x.shape)
        lateral.append(x)
        # stage 3
        x = self.layer2_flow(x) # 16x28x28
        # print(x.shape)
        lateral.append(x)
        # stage 4
        # x = self.layer3_flow(x) # 16x14x14
        # print(x.shape)
        # lateral.append(x)
        # stage 5
        # x = self.layer4_flow(x) # 16x7x7
        # print(x.shape)
        # lateral.append(x)

        return lateral
    
    
    # -------------------------------------------------------------------------
    def SlowPath(self, input, lateral):
        # stage 1
        x = self.conv1_s(input)
        x = self.conv1_t(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print('---------------slow------------')
        # print(x.shape) # 24
        # x = torch.cat([x, lateral[0]],dim=1)
        # print(x.shape) # 27
        # stage 2 ------ PROBLEMS HERE??????? # OF INPUT CHANNELS: (X + LATERAL)!!! solved, parameterize the numbers!
        x = self.layer1(x)
        # print(x.shape) # 24
        
        # x = torch.cat([x, lateral[1]],dim=1)
        # print(x.shape) # 27
        
        x = self.layer2(x)
        # print(x.shape) # 48
        x = torch.cat([x, lateral[2]],dim=1)
        # print(x.shape) # 54
        
        x = self.layer3(x)
        # print(x.shape) # 96
        # x = torch.cat([x, lateral[3]],dim=1)
        # print(x.shape) # 108
        
        x = self.layer4(x)
        # print(x.shape) # 192
        # x = torch.cat([x, lateral[4]],dim=1)
        # print(x.shape) # 216
    
        return x
    
    def forward(self, x_video, x_flow):
        lateral = self.FlowPath(x_flow)
        # print(fast.shape)
        x = self.SlowPath(x_video, lateral)
        # print(slow.shape)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.avgpool(x)
        # print(x.shape)

        x = self.fc1(x)
        x = self.relu(x)
        # print(x.shape)

        x = x.squeeze(4).squeeze(3).squeeze(2) # B C
        x = self.dropout(x)
        # print(x.shape)
        x = self.fc2(x)
        
        return x
        
        
        
# -----------------------------------------------------------------------------
        
    # Bottleneck, # planes: (54,24) #blocks:3 # shortcut_type = 'B', # stride = 2
    def _make_layer(self, block, planes, blocks, stride, in_planes_slow):
        '''
        Parameters
        ----------
        block : Bottleneck: 1 residual block, shown in above class Bottleneck method
        planes : block_inplanes[i]: input channels/output channels (res2: 54,24)
        blocks :  layers[i], how many times block is repeated (res2: 3)
        shortcut_type : 'B' ????????????
        stride : 2
        -------
        '''
        downsample = None
        
        # print(stride)
        # print(self.in_planes)
        # print(planes[1])
        
        '''
        stride != 1: 
            for example, the output of conv1 is 16x112x112, it we want to have a skip connection to the 1x1^2, 24 layer
            of the res2 block, then we need to set the stride = 2, which reduces the output size by half, and this can 
            match the output of the 1x1^2, 24 layer from res2 block
        
        in_planes != planes[1]: 
            when doing a skip connection from res2 to res3, although in_planes = 24, while planes = 48, meaning we are jumping to
            another block, and the size of image needs to be reduced by half, then downsample is applied
        '''
        if stride != 1 or in_planes_slow != planes[1]: #planes: (54,24), plane[1] = 24/48/96/192

                downsample = nn.Sequential(
                    conv1x1x1(in_planes_slow, planes[1], stride),
                    nn.BatchNorm3d(planes[1], affine=True, track_running_stats=True)
                    )
                # print(downsample)

        layers = []
        
        # On each stage, only first block need to consider the downsample problem, 
        # the rest of blocks can be appended without downsampling
        layers.append(
            block(in_planes=in_planes_slow, #27
                  planes=planes, # (54,24)
                  stride=stride, #2
                  downsample=downsample, # None/a layer
                  index=self.index, # 0
                  )) # 1
        
        in_planes_slow = planes[1] # 24, 48, 96, 192
        self.index += 1 # 1
        for i in range(1, blocks): # 3,5,11,7 number of repeated blocks
            layers.append(block(in_planes_slow, planes, index=self.index))
            self.index += 1 #2

        self.index = 0 # used for squeeze-and-excitation
        return nn.Sequential(*layers)

# -----------------------------------------------------------------------------
    def _make_layer_flow(self, block, planes, blocks,  stride):

        downsample = None

        # print(stride)
        # print(self.in_planes)
        # print(planes[1])
        if stride != 1 or self.in_planes_flow != planes[1]: #planes: (54,24), plane[1] = 24/48/96/192

                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes_flow, planes[1], stride),
                    nn.BatchNorm3d(planes[1], affine=True, track_running_stats=True)
                    )
                # print(downsample)

        layers = []
        
        # On each stage, only first block need to consider the downsample problem, 
        # the rest of blocks can be appended without downsampling
        layers.append(
            block(in_planes=self.in_planes_flow, #24
                  planes=planes, # (54,24)
                  stride=stride, #2
                  downsample=downsample, # None/a layer
                  index=self.index, # 0
                  )) # 1
        
        self.in_planes_flow = planes[1] # 24, 48, 96, 192
        self.index += 1 # 1
        for i in range(1, blocks): # 3,5,11,7 number of repeated blocks
            layers.append(block(self.in_planes_flow, planes, index=self.index))
            self.index += 1 #2

        self.index = 0 # used for squeeze-and-excitation
        return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def get_blocks(version):
    blocks = {'S':[3,5,11,7],
              'M':[3,5,11,7],
              'L': [5,10,25,15],
              'XL':[5,10,25,15]}
    return blocks[version]

def get_blocks_flow(version):
    blocks = {'S':[1,1,1,1],
              'M':[2,1,2,2],
              'L': [1,1,1,1],
              'XL':[1,1,1,1]}
    return blocks[version]

#-----------------------------------------------------------------------------
def get_inplanes(version):
    planes = {'S':[(54,24), (108,48), (216,96), (432,192)],
              'M':[(54,24), (108,48), (216,96), (432,192)],
              'L': [(54,24), (108,48), (216,96), (432,192)],
              'XL':[(72,32), (162,72), (306,136), (630,280)]}
    return planes[version]

'''
chan_ratio: x3d out channel/flow out channel = chan_ratio 
used to control the # of channels of the flow branch
'''
def get_inflows(version, chan_ratio):
    planes = {'S':[(54//chan_ratio,24//chan_ratio), (108//chan_ratio,48//chan_ratio), (216//chan_ratio,96//chan_ratio), (432//chan_ratio,192//chan_ratio)],
              'M':[(54//chan_ratio,24//chan_ratio), (108//chan_ratio,48//chan_ratio), (216//chan_ratio,96//chan_ratio), (432//chan_ratio,192//chan_ratio)],
              'L': [(54//chan_ratio,24//chan_ratio), (108//chan_ratio,48//chan_ratio), (216//chan_ratio,96//chan_ratio), (432//chan_ratio,192//chan_ratio)],
              'XL':[(72//chan_ratio,32//chan_ratio), (162//chan_ratio,72//chan_ratio), (306//chan_ratio,136//chan_ratio), (630//chan_ratio,280//chan_ratio)]}
    return planes[version]
#-----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def generate_model(x3d_version, flow_blocks, **kwargs):
    # print(kwargs)
    model = ResNet(Bottleneck, 
                   layers = get_blocks(x3d_version),  # number of layers repeated at each x3d block
                   layers_flow = flow_blocks, # number of layers repeated at each flow block
                   block_inplanes=get_inplanes(x3d_version), 
                   block_inflows = get_inflows(x3d_version, chan_ratio = 8), 
                   **kwargs)
    return model


if __name__ == "__main__":

    input_tensor1 = torch.autograd.Variable(torch.rand(1, 3, 16, 256, 256)) # [B (base_bn_splits), C, T, W, H]
    input_tensor2 = torch.autograd.Variable(torch.rand(1, 2, 16, 256, 256))
    
    input_tensor1 = input_tensor1.cuda()
    input_tensor2 = input_tensor2.cuda()
    
    # input_tensor = [input_tensor1, input_tensor2]
    # input_tensor = input_tensor.cuda()
    flow_blocks = [[1,1,1,1], [2,1,1,1], [3,1,1,1], [3,2,1,1], [3,3,1,1], [3,4,1,1]]
    
    x3d_version = 'M'
    model = generate_model(x3d_version, flow_blocks = flow_blocks[0], n_classes=400, dropout=0.5)
    model = model.cuda()
    
    
    output = model(input_tensor1, input_tensor2)
    # print(input_tensor.shape)
    print(output.shape)
    
    # print(torch.max(output))
    # print(torch.argmax(output))
    
    
    # model_path = 'X3D-M-LATERAL-ONE.pt' # lateral connections only after conv1
    # model_path = 'X3D-M-LATERAL-TWO.pt' # lateral connections only after res1 
    model_path = 'X3D-M-LATERAL-THREE.pt' # lateral connections only after res2
    load_ckpt = torch.load(model_path)
    
    model.load_state_dict(load_ckpt, strict = False)
    
    
    

    
