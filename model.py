import xml.etree.ElementTree as ET
import os as os
from PIL import Image
import numpy as np
import torch
import os
from glob import glob
import cv2
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch import randperm
from torch._utils import _accumulate
import torchvision
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as F
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import sklearn.metrics as skm
from scipy.spatial import distance
from scipy.stats import multivariate_normal
from torch.optim import Adam

class NimbRoNet2(nn.Module):
    def __init__(self):
        super(NimbRoNet2, self).__init__()
        self.encoder = models.resnet18(pretrained=True)

        self.conv1x1_1 = self.conv_block(64, 128, 1)
        self.conv1x1_2 = self.conv_block(128, 256, 1)
        self.conv1x1_3 = self.conv_block(256, 256, 1)

        self.transpose_conv1 = self.transpose_conv_block(512,256,2,2, BatchNorm=False, Activation=True)
        self.transpose_conv2 = self.transpose_conv_block(512,256,2,2, BatchNorm=True, Activation=True)
        self.transpose_conv3 = self.transpose_conv_block(512,128,2,2, BatchNorm=True, Activation=True)

        self.relu = nn.ReLU(False)
        self.bn = nn.BatchNorm2d(256)

        self.location_bias = torch.nn.Parameter(torch.zeros(120,160,3))
        self.loc_dep_conv = LocationAwareConv2d(self.location_bias,True,120,160,256,3,1)
        self.segmentation_conv = LocationAwareConv2d(self.location_bias,True,120,160,256,3,1)


    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x_layer1 = self.encoder.layer1(x)
        x_layer2 = self.encoder.layer2(x_layer1)
        x_layer3 = self.encoder.layer3(x_layer2)
        x_layer4 = self.encoder.layer4(x_layer3)

        x_layer1 = self.conv1x1_1(x_layer1)
        x_layer2 = self.conv1x1_2(x_layer2)
        x_layer3 = self.conv1x1_3(x_layer3)

        x_layer4 = self.transpose_conv1(x_layer4)

        x_layer3_4 = torch.cat((x_layer3,x_layer4),dim=1)
        x_layer3_4 = self.transpose_conv2(x_layer3_4)

        x_layer3_4_2 = torch.cat((x_layer3_4,x_layer2),dim=1)
        x_layer3_4_2 = self.transpose_conv3(x_layer3_4_2)

        x_layer3_4_2_1 = torch.cat((x_layer3_4_2,x_layer1),dim=1)

        x_layer3_4_2_1 = self.relu(x_layer3_4_2_1)
        x_layer3_4_2_1 = self.bn(x_layer3_4_2_1)

        detection = self.loc_dep_conv(x_layer3_4_2_1)
        segmentation = self.segmentation_conv(x_layer3_4_2_1)

        return detection, segmentation


    def transpose_conv_block(self, in_f, out_f, kernel, stride, BatchNorm=True, Activation=True):
        modules = []
        if Activation:
            modules.append(nn.ReLU(False))
        if BatchNorm:
            modules.append(nn.BatchNorm2d(in_f))
        modules.append(nn.ConvTranspose2d(in_f, out_f, kernel, stride))
        return nn.Sequential(*modules)
    def conv_block(self, in_f, out_f, kernel):
        return nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel)
        )

class LocationAwareConv2d(torch.nn.Conv2d):
    def __init__(self,location_bias,gradient,w,h,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.locationBias=location_bias
        self.locationEncode=torch.autograd.Variable(torch.ones(w,h,3))
        if gradient:
            for i in range(w):
                self.locationEncode[i,:,1]=self.locationEncode[:,i,0]=(i/float(w-1))
    def forward(self,inputs):
        if self.locationBias.device != inputs.device:
            self.locationBias=self.locationBias.to(inputs.get_device())
        if self.locationEncode.device != inputs.device:
            self.locationEncode=self.locationEncode.to(inputs.get_device())
        b=self.locationBias*self.locationEncode
        return super().forward(inputs)+b[:,:,0]+b[:,:,1]+b[:,:,2]
