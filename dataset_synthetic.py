import os
import random
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, root_path, transform, pose_code):
        self.root_path = root_path
        self.transform = transform
    
        self.gallery_list = []
        self.gallery_labels = []
        self.poses = {
            '15': ('140', '050'),
            '90': ('110', '240'),
            '75': ('120', '010'),
            '60': ('090', '200'),
            '45': ('080', '190'),
            '30': ('130', '041'),

        }
        pos, neg = self.poses[pose_code][0], self.poses[pose_code][1]
        for j in os.listdir(self.root_path):
            # dir_path = os.path.join(self.root_path, i)
            # if os.path.isdir(dir_path):
            # content = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
            # for j in os.listdir(dir_path):
            img_pose = os.path.basename(j).split('_')[3]
            id_label = os.path.basename(j).split('_')[0]
            
            if (img_pose == pos or img_pose == neg) and os.path.basename(j).split('_')[-1].startswith('fake'):
                self.gallery_list.append(os.path.join(self.root_path, j))
                self.gallery_labels.append(id_label)
        
        self.data_len = len(self.gallery_labels)
    

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        img = Image.open(self.gallery_list[idx])
        img = self.transform(img)
        label = self.gallery_labels[idx]

        return img, label