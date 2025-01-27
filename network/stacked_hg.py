import torch.nn as nn
import torch.nn.functional as F

import wandb
import matplotlib.pyplot as plt
from termcolor import colored
import numpy as np
import cv2
import ipdb


class Hourglass(nn.Module):
    def __init__(self, res_block, num_blocks, inplanes, depth):
        super().__init__()
        self.depth = depth

        self.encoders = self._make_encoder(res_block, num_blocks, inplanes, depth)
        self.decoders = self._make_decoder(res_block, num_blocks, inplanes, depth)

    def _make_residual(self, block, num_blocks, inplanes):
        layers = [block(inplanes) for _ in range(num_blocks)]

        return nn.Sequential(*layers)

    def _make_encoder(self, block, num_blocks, inplanes, depth):
        encoders = []
        for i in range(depth):
            res = [self._make_residual(block, num_blocks, inplanes) for _ in range(2)]
            if i == depth - 1:
                res.append(self._make_residual(block, num_blocks, inplanes))
            encoders.append(nn.ModuleList(res))

        return nn.ModuleList(encoders)

    def _make_decoder(self, block, num_blocks, inplanes, depth):
        decoders = [self._make_residual(block, num_blocks, inplanes) for _ in range(depth)]

        return nn.ModuleList(decoders)

    def _encoder_forward(self, x):
        out = []
        for i in range(self.depth):
            out.append(self.encoders[i][0](x))
            x = self.encoders[i][1](F.max_pool2d(x, 2, stride=2))

            if i == self.depth - 1:
                out.append(self.encoders[i][2](x))

        return out[::-1]

    def _decoder_forward(self, x):
        out = x[0]

        for i in range(self.depth):
            up = x[i + 1]
            low = self.decoders[i](out)
            low = F.interpolate(low, scale_factor=2)
            out = low + up

        return out

    def forward(self, x):
        x = self._encoder_forward(x)  # 5个值的list， 8， 16， 32， 64， 128
        out = self._decoder_forward(x)
        
        return out


class HourglassNet(nn.Module):
    def __init__(self, use_cbam , res_block, head_block, in_channels, inplanes, num_feats, depth, num_stacks, num_blocks, num_classes):
        super().__init__()
        self.use_cbam = use_cbam
        self.num_stacks = num_stacks

        # Shallow feature extraction and modulations
        # res_block = ResNetBottleneck  inplanes: 64   num_feats: 256
        self.shallow_conv, self.shallow_res = self._make_shallow_layer(res_block, in_channels, inplanes, num_feats)

        # Hourglass modules
        self.hgs = nn.ModuleList([Hourglass(res_block, num_blocks, num_feats, depth) for _ in range(num_stacks)])
        self.res = nn.ModuleList([self._make_residual(res_block, num_blocks, num_feats) for _ in range(num_stacks)])
        self.fcs = nn.ModuleList([self._make_fc(num_feats, num_feats) for _ in range(num_stacks)])
        self.scores = nn.ModuleList([head_block(num_feats, num_classes) for _ in range(num_stacks)])

        self.fcs_ = nn.ModuleList([nn.Conv2d(num_feats, num_feats, 1) for _ in range(num_stacks - 1)])
        self.scores_ = nn.ModuleList([nn.Conv2d(num_classes, num_feats, 1) for _ in range(num_stacks - 1)])

    def _make_residual(self, block, num_blocks, inplanes, planes=None, stride=1):
        planes = planes or inplanes // block.expansion
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Conv2d(inplanes, planes * block.expansion, 1, stride=stride)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_shallow_layer(self, block, in_channels, inplanes, num_feats):
        shallow_conv = nn.Sequential(
            nn.Conv2d(in_channels, inplanes, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True)
        )

        shallow_res = nn.Sequential(
            self._make_residual(block, 1, inplanes, inplanes),
            nn.MaxPool2d(2, stride=2),
            self._make_residual(block, 1, inplanes * block.expansion, inplanes * block.expansion),
            self._make_residual(block, 1, inplanes * block.expansion ** 2, num_feats // block.expansion)
        )

        return shallow_conv, shallow_res

    def _make_fc(self, inplanes, outplanes):
        fc = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )

        return fc

    def forward(self, x):
        out = []
        x = self.shallow_conv(x) 
        x = self.shallow_res(x)
        feature_list = []
        for i in range(self.num_stacks):
            y= self.hgs[i](x)
            y = self.fcs[i](self.res[i](y)) # bs, 256, 128, 128

            score = self.scores[i](y)
            out.append(score)

            
            if i < self.num_stacks - 1:
                score_ = self.scores_[i](score)
                y = self.fcs_[i](y)
                x = x + y + score_

        return out[::-1], y 
