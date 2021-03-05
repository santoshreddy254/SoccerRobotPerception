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
import numbers
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as get_data_loader
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

from dataloader import Blobdataset, SegDataset, split_dataset
from model import NimbRoNet2
from train import train_model

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
device = torch.device("cpu")
print("Using device: ", device)


batch_size = 2
num_epochs = 1
print("======Loading detection dataset======")
detection_dataset = Blobdataset("/home/santosh/MAS/CUDA_vision/Project/data/blob")
train_detection_dataset, val_detection_dataset, test_detection_dataset = split_dataset(detection_dataset)
print("Total samples: ",len(detection_dataset))
print("Training samples: ",len(train_detection_dataset))
print("Validation samples: ",len(val_detection_dataset))
print("Testing samples: ",len(test_detection_dataset))
print("=====================================")

print("======Loading segmentation dataset======")
segmentation_dataset = SegDataset("/home/santosh/MAS/CUDA_vision/Project/data/segmentation")
train_segmentation_dataset, val_segmentation_dataset, test_segmentation_dataset = split_dataset(segmentation_dataset)
print("Total samples: ",len(segmentation_dataset))
print("Training samples: ",len(train_segmentation_dataset))
print("Validation samples: ",len(val_segmentation_dataset))
print("Testing samples: ",len(test_segmentation_dataset))
print("========================================")

train_detection_loader = get_data_loader(dataset=train_detection_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

train_seg_loader = get_data_loader(dataset=train_segmentation_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_detection_loader = get_data_loader(dataset=val_detection_dataset,
                                           batch_size=1,
                                           shuffle=True)

val_seg_loader = get_data_loader(dataset=val_segmentation_dataset,
                                           batch_size=1,
                                           shuffle=True)

test_detection_loader = get_data_loader(dataset=test_detection_dataset,
                                           batch_size=1,
                                           shuffle=True)

test_seg_loader = get_data_loader(dataset=test_segmentation_dataset,
                                           batch_size=1,
                                           shuffle=True)

print("======Loading NimbRoNet2 model======")
model = NimbRoNet2()

optimizer = Adam([{"params":model.encoder.parameters(), "lr":0.000001},
                {"params":model.conv1x1_1.parameters(), "lr":0.001},
                {"params":model.conv1x1_2.parameters(), "lr":0.001},
                {"params":model.conv1x1_3.parameters(), "lr":0.001},
                {"params":model.transpose_conv1.parameters(), "lr":0.001},
                {"params":model.transpose_conv2.parameters(), "lr":0.001},
                {"params":model.transpose_conv3.parameters(), "lr":0.001},
                {"params":model.loc_dep_conv.parameters(), "lr":0.001}],
                lr=0.001)

print("======Training NimbRoNet2 model started======")
train_model(model, num_epochs, train_detection_loader, train_seg_loader, val_detection_loader, val_seg_loader, optimizer, device)
