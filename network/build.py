from network.stacked_hg import HourglassNet
from network.multi_task_head import MultiTaskHead
import torch
import torch.nn as nn

import ipdb


class ResNetBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes=None, stride=1, downsample=None):
        super().__init__()
        planes = planes or inplanes // self.expansion

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        return out


def build_hg(cfg, in_channels):
    inplanes = cfg.inplanes
    num_feats = cfg.num_feats
    depth = cfg.depth
    num_stacks = cfg.num_stacks
    num_blocks = cfg.num_blocks
    head_size = cfg.head_size


    num_classes = sum(head_size)
    model = HourglassNet(
        use_cbam = False,
        
        res_block=ResNetBottleneck,
        head_block=lambda c_in, c_out: MultiTaskHead(c_in, c_out, head_size=head_size),
        in_channels=in_channels,
        inplanes=inplanes,
        num_feats=num_feats,
        depth=depth,
        num_stacks=num_stacks,
        num_blocks=num_blocks,
        num_classes=num_classes
    )

    return model
