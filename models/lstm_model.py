import argparse
import math
import random
import os,time
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
import torchvision
from torchvision import transforms, utils

import torch.nn as nn

import itertools
from .encoders.psp_encoders import *
from .stylegan2.model import *
from .ConvLSTM import ConvLSTM
from models.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE, _upsample_add






class pSp_LSTM_Encoder(Module):
    def __init__(self, num_layers=50, input_channel=9, mode='ir'):
        super(pSp_LSTM_Encoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(input_channel, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()

        log_size = int(math.log(1024, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7

        self.convlstm1 = ConvLSTM(input_size=512, hidden_size=512, kernel_size=3)
        self.convlstm2 = ConvLSTM(input_size=512, hidden_size=512, kernel_size=3)


        for i in range(self.middle_ind):

            
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            else:
                style = GradualStyleBlock(512, 512, 32)

            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, x, prev_state1, prev_state2):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            # if i == 6:
            #     c1 = x
            if i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        state1 = self.convlstm1(c3, prev_state1)
        c3 = state1[0]

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = _upsample_add(c3, self.latlayer1(c2))
        
        state2 = self.convlstm2(p2, prev_state2)
        p2 = state2[0]

        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        out = torch.stack(latents, dim=1)
        return out,state1,state2






class E4E_LSTM_Encoder(Module):
    def __init__(self, num_layers=50, input_channel=9, mode='ir'):
        super(E4E_LSTM_Encoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(input_channel, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.styles = nn.ModuleList()


        self.convlstm = ConvLSTM(input_size=512, hidden_size=512, kernel_size=3)
        
        self.style = GradualStyleBlock(512, 512, 16)

    def forward(self, x, prev_state):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 23:
                c = x

        state = self.convlstm(c, prev_state)
        out = self.style(state[0])
        return out,state








class GPEN_LSTM_Encoder(nn.Module):
    def __init__(
        self,
        input_channel=9,
        size=1024,
        channel_multiplier=2,
        narrow=1,
    ):
        super().__init__()
        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow)
        }

        self.log_size = int(math.log(size, 2))        
        conv = [ConvLayer(input_channel, channels[size], 1)]
        self.ecd0 = nn.Sequential(*conv)
        in_channel = channels[size]

        self.names = ['ecd%d'%i for i in range(self.log_size-1)]
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            conv = [ConvLayer(in_channel, out_channel, 3, downsample=True)] 
            setattr(self, self.names[self.log_size-i+1], nn.Sequential(*conv))
            in_channel = out_channel
        # self.final_linear = nn.Sequential(EqualLinear(channels[4]*4*4, 14*512, activation='fused_lrelu'))
        self.final_linear = nn.Sequential(EqualLinear(channels[4]*4*4, 10*512, activation='fused_lrelu'))

        self.convlstm = ConvLSTM(input_size=512, hidden_size=512, kernel_size=3)

    def forward(self,
        inputs,prev_state
    ):
        for i in range(self.log_size-1):
            ecd = getattr(self, self.names[i])
            inputs = ecd(inputs)

        state = self.convlstm(inputs, prev_state)
        offset = state[0].view(inputs.shape[0], -1)
        # offset = self.final_linear(offset).reshape(-1,14,512)
        offset = self.final_linear(offset).reshape(-1,10,512)


        return offset,state






class HIFI_LSTM_Encoder(Module):
    def __init__(self, input_channel=9):
        super(HIFI_LSTM_Encoder, self).__init__()
        self.conv_layer1 = Sequential(Conv2d(input_channel, 32, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(32),
                                      PReLU(32))

        self.conv_layer2 =  Sequential(*[bottleneck_IR(32,48,2), bottleneck_IR(48,48,1), bottleneck_IR(48,48,1)])

        self.conv_layer3 =  Sequential(*[bottleneck_IR(48,64,2), bottleneck_IR(64,64,1), bottleneck_IR(64,64,1)])

        self.condition_scale3 = nn.Sequential(
                    EqualConv2d(64, 512, 3, stride=1, padding=1, bias=True ),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(512, 512, 3, stride=1, padding=1, bias=True ))

        self.condition_shift3 = nn.Sequential(
                    EqualConv2d(64, 512, 3, stride=1, padding=1, bias=True ),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(512, 512, 3, stride=1, padding=1, bias=True ))  

        #self.final_linear = nn.Sequential(EqualLinear(64*64*64, 7*512, activation='fused_lrelu'))
        self.convlstm = ConvLSTM(input_size=64, hidden_size=64, kernel_size=3)


    def forward(self, x, prev_state):
        conditions = []
        feat = self.conv_layer1(x)
        feat = self.conv_layer2(feat)
        feat = self.conv_layer3(feat)

        state = self.convlstm(feat,prev_state)
        feat = state[0]

        scale = self.condition_scale3(feat)
        scale = torch.nn.functional.interpolate(scale, size=(64,64) , mode='bilinear')
        conditions.append(scale.clone())
        shift = self.condition_shift3(feat)
        shift = torch.nn.functional.interpolate(shift, size=(64,64) , mode='bilinear')
        conditions.append(shift.clone()) 

        #offset = self.final_linear(feat.view(x.shape[0], -1)).reshape(-1,7,512)

        # return offset,conditions,state
        return conditions,state


        




class GPEN_Noise_Encoder(nn.Module):
    def __init__(
        self,
        input_channel=9,
        size=1024,
        channel_multiplier=2,
        narrow=1,
    ):
        super().__init__()
        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow)
        }

        self.log_size = int(math.log(size, 2))        
        conv = [ConvLayer(input_channel, channels[size], 1)]
        self.ecd0 = nn.Sequential(*conv)
        in_channel = channels[size]

        self.names = ['ecd%d'%i for i in range(self.log_size-1)]
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            conv = [ConvLayer(in_channel, out_channel, 3, downsample=True)] 
            setattr(self, self.names[self.log_size-i+1], nn.Sequential(*conv))
            in_channel = out_channel

    def forward(self,
        inputs,noise_res
    ):
        noises = []
        
        for i in range(self.log_size-1):
            ecd = getattr(self, self.names[i])
            inputs = ecd(inputs)
            if inputs.shape[2]<=noise_res:
                noises.append(inputs)
            else:
                noises.append(torch.zeros_like(inputs))

        noises = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in noises))[::-1][1:]
        return noises





class GPEN_LSTM_Noise_Encoder_Selected(nn.Module):
    def __init__(
        self,
        input_channel=9,
        size=1024,
        channel_multiplier=2,
        narrow=1,
    ):
        super().__init__()
        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow)
        }

        self.log_size = int(math.log(size, 2))        
        conv = [ConvLayer(input_channel, channels[size], 1)]
        self.ecd0 = nn.Sequential(*conv)
        in_channel = channels[size]

        self.names = ['ecd%d'%i for i in range(self.log_size-1)]
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            conv = [ConvLayer(in_channel, out_channel, 3, downsample=True)] 
            setattr(self, self.names[self.log_size-i+1], nn.Sequential(*conv))
            in_channel = out_channel
        self.final_linear = nn.Sequential(EqualLinear(channels[4]*4*4, 10*512, activation='fused_lrelu'))
        # self.final_linear = nn.Sequential(EqualLinear(channels[4]*4*4, 14*512, activation='fused_lrelu'))

        self.convlstm = ConvLSTM(input_size=512, hidden_size=512, kernel_size=3)

    def forward(self,
        inputs,prev_state,noise_res
    ):
        
        for i in range(self.log_size-1):
            ecd = getattr(self, self.names[i])
            inputs = ecd(inputs)
            if inputs.shape[2]==noise_res:
                noise = inputs

        state = self.convlstm(inputs, prev_state)
        offset = state[0].view(inputs.shape[0], -1)
        # offset = self.final_linear(offset).reshape(-1,14,512)
        offset = self.final_linear(offset).reshape(-1,10,512)


        return offset,state,noise




class GPEN_Noise_Encoder_Selected(nn.Module):
    def __init__(
        self,
        input_channel=9,
        size=1024,
        channel_multiplier=2,
        narrow=1,
    ):
        super().__init__()
        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow)
        }

        self.log_size = int(math.log(size, 2))        
        conv = [ConvLayer(input_channel, channels[size], 1)]
        self.ecd0 = nn.Sequential(*conv)
        in_channel = channels[size]

        self.names = ['ecd%d'%i for i in range(self.log_size-1)]
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            conv = [ConvLayer(in_channel, out_channel, 3, downsample=True)] 
            setattr(self, self.names[self.log_size-i+1], nn.Sequential(*conv))
            in_channel = out_channel
        self.final_linear = nn.Sequential(EqualLinear(channels[4]*4*4, 10*512, activation='fused_lrelu'))
        #self.convlstm = ConvLSTM(input_size=512, hidden_size=512, kernel_size=3)

    def forward(self,
        inputs,noise_res
    ):
        
        for i in range(self.log_size-1):
            ecd = getattr(self, self.names[i])
            inputs = ecd(inputs)
            if inputs.shape[2]==noise_res:
                noise = inputs

        # state = self.convlstm(inputs, prev_state)
        offset = inputs.view(inputs.shape[0], -1)
        offset = self.final_linear(offset).reshape(-1,10,512)

        return offset,noise



class GPEN_LSTM_Noise_Encoder_Full(nn.Module):
    def __init__(
        self,
        input_channel=9,
        size=1024,
        channel_multiplier=2,
        narrow=1,
    ):
        super().__init__()
        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow)
        }

        self.log_size = int(math.log(size, 2))        
        conv = [ConvLayer(input_channel, channels[size], 1)]
        self.ecd0 = nn.Sequential(*conv)
        in_channel = channels[size]

        self.names = ['ecd%d'%i for i in range(self.log_size-1)]
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            conv = [ConvLayer(in_channel, out_channel, 3, downsample=True)] 
            setattr(self, self.names[self.log_size-i+1], nn.Sequential(*conv))
            in_channel = out_channel
        self.final_linear = nn.Sequential(EqualLinear(channels[4]*4*4, 10*512, activation='fused_lrelu'))
        self.convlstm = ConvLSTM(input_size=512, hidden_size=512, kernel_size=3)

    def forward(self,
        inputs,prev_state,noise_res
    ):
        
        noises = []
        
        for i in range(self.log_size-1):
            ecd = getattr(self, self.names[i])
            inputs = ecd(inputs)
            if inputs.shape[2]<=noise_res:
                noises.append(inputs)
            else:
                noises.append(torch.zeros_like(inputs))

        state = self.convlstm(inputs, prev_state)
        offset = state[0].view(inputs.shape[0], -1)
        offset = self.final_linear(offset).reshape(-1,10,512)
        noises = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in noises))[::-1][1:]

        return offset,state,noises