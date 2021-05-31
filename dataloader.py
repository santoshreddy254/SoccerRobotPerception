
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


class Blobdataset(Dataset):
    def __init__(self, path, Transform=None):
        self.path = path
        self.transform = Transform
        files = []
        self.filtered_files = []
        self.tranform_input = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize((480,640)),
                                                  transforms.ToTensor()])
        self.transform_heatmap = transforms.Compose([transforms.ToPILImage(),
                                                     transforms.Resize((120,160)),
                                                     transforms.ToTensor()])
        for ext in ('**/*.jpeg', '**/*.png', '**/*.jpg'):
            files.extend(glob(os.path.join(path, ext),recursive=True))

        for file in files:
            ext = file.split(".")[-1]
            self.xml_path = file.replace("."+ext,".xml")
            if os.path.isfile(self.xml_path):
                self.filtered_files.append(file)
    def __len__(self):
        return len(self.filtered_files)

    def __getitem__(self, index):

        image_path = self.filtered_files[index]
        ext = image_path.split(".")[-1]
        xml_path = image_path.replace("."+ext,".xml")
        annotation = ET.parse(xml_path).getroot()
        image = F.to_tensor(Image.open(image_path))


#         if(self.transform):

#             if(type(self.transform) is not list):
#                 self.transform = [self.transform]

#             for idx in range(len(self.transform)):
#                 data_dict = self.transform[idx](data_dict)

        image_size = image.shape
        heatmap_placeholder = torch.ones([3, int(image_size[1]/4), int(image_size[2]/4)])

        class_flags = [False,False,False]
        for obj in annotation.findall("object"):

            label = obj.find("name").text
            anno_bound_box = obj.find("bndbox")
            bound_box_center = self.get_center(anno_bound_box)


            if(label == "ball"):
                class_flags[0] = True
                heatmap_placeholder[0] -= self.get_heatmap(image_size,bound_box_center,3)
            elif (label == "goalpost"):
                class_flags[1] = True
                heatmap_placeholder[1] -= self.get_heatmap(image_size,bound_box_center,3)
            elif (label == "robot"):
                class_flags[2] = True
                heatmap_placeholder[2] -= self.get_heatmap(image_size,bound_box_center,10)

        for i in range(heatmap_placeholder.shape[0]):
            heatmap_placeholder[i] = self.get_normalized_heatmap(heatmap_placeholder[i],class_flags[i])

        return self.tranform_input(image), self.transform_heatmap(heatmap_placeholder)
    def get_normalized_heatmap(self,heatmap_single_channel,class_flag):
        n = (heatmap_single_channel-torch.min(heatmap_single_channel))
        d = (torch.max(heatmap_single_channel)-torch.min(heatmap_single_channel))
        if class_flag:
            return n/d
        else:
            return heatmap_single_channel
    def get_center(self,bound_box):
        xmin = int(bound_box.find('xmin').text)
        ymin = int(bound_box.find('ymin').text)
        xmax = int(bound_box.find('xmax').text)
        ymax = int(bound_box.find('ymax').text)
        return (int((xmin+xmax)/2),int((ymin+ymax)/2))
    def get_heatmap(self,image_size,center_point,variance):
        new_height, new_width = int(image_size[1]/4), int(image_size[2]/4)
        #https://stackoverflow.com/questions/54726703/generating-keypoint-heatmaps-in-tensorflow
        mean = [center_point[1]/4,center_point[0]/4]
        pos = np.dstack(np.mgrid[0:new_height:1, 0:new_width:1])
        rv = multivariate_normal(mean, cov=variance)
        return variance*rv.pdf(pos).reshape(new_height,new_width)


class SegDataset(Dataset):

    def __init__(self, path, Transform = None):

        self.path = path
        self.transform = Transform
        self.image_files = []
        self.tranform_input = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize((480,640)),
                                                  transforms.ToTensor()])
        self.transform_heatmap = transforms.Compose([transforms.ToPILImage(),
                                                     transforms.Resize((120,160)),
                                                     transforms.ToTensor()])
        for ext in ('**/image/*.jpeg', '**/image/*.png', '**/image/*.jpg'):
            self.image_files.extend(glob(os.path.join(path, ext),recursive=True))

    def __len__(self):

        return len(self.image_files)

    def load_image(self, path):
        raw_image = Image.open(path).convert('RGB')
        raw_image = F.resize(raw_image, (480, 640))
        norm_image = np.array(raw_image, dtype=np.float32)/255.0
        img_rearrange = np.transpose(norm_image, (2, 0, 1))
        return img_rearrange

    def load_mask(self, path):
        raw_image = Image.open(path)
        raw_image = np.array(raw_image)

        raw_image[raw_image == 1] = 3    #change ball to ground
        raw_image[raw_image == 2] = 2
        raw_image[raw_image == 3] = 1

        img_resized = cv2.resize(raw_image, dsize=(160, 120), interpolation=cv2.INTER_NEAREST)
        img_array = np.array(img_resized)
        img_array = np.transpose(img_array, (0, 1))
        return img_array

    def __getitem__(self, index):

        image_path = self.image_files[index]
        target_path = image_path.replace("/image/","/target/")
        ext = image_path.split(".")[-1]
        target_path = target_path.replace("."+ext,".png")

        image = torch.FloatTensor(self.load_image(image_path))
        target = torch.FloatTensor(self.load_mask(target_path))

        return self.tranform_input(image), target


def split_dataset(dataset):
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    torch.manual_seed(0)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset
