import torch.utils.data 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import numpy as np
import cv2

data_mapping = {
  'level0': 0,
  'level1': 1,
  'level2': 2,
  'level3': 3,
  'level4': 4,
}

class TrainData(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        super(TrainData).__init__()
        self.transform = transforms.Compose([transforms.Resize((400, 400)),
                                transforms.RandomHorizontalFlip(p=1),
                                transforms.ToTensor()]) 
        self.base_dir = os.path.join(data_dir, "train")
        self.list_image = glob.glob(os.path.join(self.base_dir, '*', '*'))
        self.len = len(self.list_image)

    def __getitem__(self, index):    
        img_file = self.list_image[index]
        img = Image.open(img_file)
        img = self.transform(img)

        haze_level = img_file.split(os.sep)[-2]
        label = data_mapping[haze_level]
        return img, label

    def __len__(self):
        return self.len

class ValidData(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        super(ValidData).__init__()
        self.transform = transforms.Compose([transforms.Resize((400, 400)), transforms.ToTensor()]) 
        self.base_dir = os.path.join(data_dir, "valid")
        self.list_image = glob.glob(os.path.join(self.base_dir, '*', '*'))
        self.len = len(self.list_image)

    def __getitem__(self, index):    
        img_file = self.list_image[index]
        img = Image.open(img_file)
        img = self.transform(img)

        haze_level = img_file.split(os.sep)[-2]
        label = data_mapping[haze_level]
        return img, label
    
    def __len__(self):
        return self.len 

class TestData(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        super(TestData).__init__()
        self.transform = transforms.Compose([transforms.Resize((400, 400)), transforms.ToTensor()]) 
        self.base_dir = os.path.join(data_dir, "test")
        self.list_image = glob.glob(os.path.join(self.base_dir, '*', '*'))
        self.len = len(self.list_image)

    def __getitem__(self, index):    
        img_file = self.list_image[index]
        img = Image.open(img_file)
        img = self.transform(img)

        haze_level = img_file.split(os.sep)[-2]
        label = data_mapping[haze_level]
        return img, label

    def __len__(self):
        return self.len 
