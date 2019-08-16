from __future__ import print_function

import os
import os.path
import numpy as np
import random
import pickle
import math
import sys
import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

# Set the appropriate paths of the datasets here.
_TIERED_IMAGENET_DATASET_DIR = '../data/tieredimagenet/'

def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data

file_path = {'train':[os.path.join(_TIERED_IMAGENET_DATASET_DIR, 'train_images.npz'), os.path.join(_TIERED_IMAGENET_DATASET_DIR, 'train_labels.pkl')],
             'val':[os.path.join(_TIERED_IMAGENET_DATASET_DIR, 'val_images.npz'), os.path.join(_TIERED_IMAGENET_DATASET_DIR,'val_labels.pkl')],
             'test':[os.path.join(_TIERED_IMAGENET_DATASET_DIR, 'test_images.npz'), os.path.join(_TIERED_IMAGENET_DATASET_DIR, 'test_labels.pkl')]}

class tieredImageNet(data.Dataset):
    def __init__(self, phase='train', data_aug = False):
        assert(phase=='train' or phase=='val' or phase=='test')
        image_path = file_path[phase][0]
        label_path = file_path[phase][1]

        data_train = load_data(label_path)
        labels = data_train['labels']
        self.data = np.load(image_path)['images']

        label = []
        lb = -1
        self.wnids = []
        for wnid in labels:
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            label.append(lb)

        self.label = label
        self.num_class = len(set(label))

        mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
        std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]
        img = self.transform(Image.fromarray(img))
        return img, label

    def __len__(self):
        return len(self.data)