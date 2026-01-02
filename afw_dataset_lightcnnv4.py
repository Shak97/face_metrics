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
        
        self.poses = list(range(1,13))
        # pos, neg = self.poses[pose_code][0], self.poses[pose_code][1]
        for i in os.listdir(self.root_path):
            dir_path = os.path.join(self.root_path, i)
            if os.path.isdir(dir_path):
                content = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
                for j in content:
                    img_pose = os.path.basename(j).split('_')[3].split('.')[0]
                    id_label = '_'.join(os.path.basename(j).split('_')[0:3])
                    if int(img_pose) == pose_code:
                        self.gallery_list.append(j)
                        self.gallery_labels.append(id_label)
        
        self.data_len = len(self.gallery_labels)
    

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        img = Image.open(self.gallery_list[idx])
        img = self.transform(img)
        label = self.gallery_labels[idx]

        return img, label