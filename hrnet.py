from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def conv3x3(inchanel,outchanel, stride=1 ):
    # (w,h,i)——>s=1,k=3,p=1——>(w,h,o)只改变通道数
    return nn.conv2d(inchanel, outchanel, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inchanel, outchanel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inchanel, outchanel, stride) #s=1,2
        self.bn1 = nn.BatchNorm2d(outchanel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outchanel, outchanel) # s=1
        self.bn2 = nn.BatchNorm2d(outchanel)
        self.downsample = downsample #id s=2, need downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)

class Bottleneck(nn.modules):
    expansion = 4

    def __init__(self, inchanel, outchanel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1= nn.Conv2d(inchanel, outchanel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchanel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inchanel, outchanel, stride) #s=1,2
        self.conv3 = conv3x3(outchanel, outchanel*self.expansion) # s=1, outchanel扩展
        self.bn2 = nn.BatchNorm2d(outchanel)
        self.bn3 = nn.BatchNorm2d(outchanel*self.expansion)
        self.downsample = downsample #if s=2, need downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)
    
class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channlels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()

    def _check_branches(self, num_branches, blocks, num_blocks, 
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)
        
    def _make_one_branch(self, branch_index, block, num_blocks, 
                         num_channels, stride=1):
        downsample = None
        
blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes= 64
        super(PoseHighResolutionNet, self).__init__()

        # 2层conv进行下采样
        self.covn1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 经过layer1 Bottleneck*4
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        # 经过transition1
        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256, num_channels])

        # 经过stage2


    def _make_transition_layer(self,
                               num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels_pre_layer[i],
                                      num_channels_cur_layer[i],
                                      3,1,1,bias=False),
                                      nn.BatchNorm2d(num_channels_cur_layer[i]),
                                      nn.ReLU(True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3 = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_branches_pre[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels,3,2,1,bias=True
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3))

            
        return nn.ModuleList(transition_layers)
                    
                    






    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes*block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []   
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)







    
    



