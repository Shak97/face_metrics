import imageio
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

# from lightcnn.lightcnn import LightCNN_29Layers_v2
from light_cnn_v4 import LightCNN_V4

import os
import  torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from torch.utils.data import DataLoader

import random


from afw_dataset_lightcnnv4 import ImageDataset

device = torch.device(1)

model = LightCNN_V4(None)
state_dict = torch.load(r'lightcnn/LightCNN-V4_checkpoint.pth.tar', map_location=device)['state_dict']
model.load_state_dict(state_dict, strict = True)
model = model.to(device)

lightcnn4_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    # transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0])
])


# case 1
afw_directory_original = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cycleGAN_w_id_w_pose/datasets/multipie/AFW_cropped/'
afw_directory_synthetic = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cycleGAN_w_id_w_pose/results_afw_second_last/multipie_pretrained/test_latest/images_root'
groundtruth_directory = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cycleGAN_w_id_w_pose/results_afw_second_last/multipie_pretrained/test_latest/images_structure'

# # cyclegan w id w pose
# afw_directory_original = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cycleGAN_w_id_w_pose/datasets/multipie/AFW_cropped/'
# afw_directory_synthetic = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/cyclegan_w_id_w_pose_loss_corrected/results_afw/multipie_pretrained/test_latest/images_root'
# groundtruth_directory = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/cyclegan_w_id_w_pose_loss_corrected/results_afw/multipie_pretrained/test_latest/images_structure'

# deformable cyclegan wo id w pose
afw_directory_original = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cycleGAN_w_id_w_pose/datasets/multipie/AFW_cropped/'
afw_directory_synthetic = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cyclegan_wo_id_w_pose_corrected/results/multipie_pretrained/test_latest/images_root'
groundtruth_directory = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cyclegan_wo_id_w_pose_corrected/results/multipie_pretrained/test_latest/images_structure'

# # deform w id wo pose
# afw_directory_original = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cycleGAN_w_id_w_pose/datasets/multipie/AFW_cropped/'
# afw_directory_synthetic = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cycleGAN_w_id/results/multipie_pretrained/test_latest/images_root'
# groundtruth_directory = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cycleGAN_w_id/results/multipie_pretrained/test_latest/images_structure'


probe_list = [os.path.join(afw_directory_synthetic, f.split('.')[0] + '_0.png') for f in os.listdir(groundtruth_directory) if f.endswith('.png')]

synthetic_content = os.listdir(afw_directory_synthetic)


probe_labels = []
probe_emb = []

with torch.no_grad():
    for probe_img in probe_list:
        img = Image.open(probe_img)
        img = lightcnn4_transform(img)[None,:,:,:]
        emb = model(img.to(device))
        # emb = emb.detach().numpy()
        probe_emb.append(emb)
        id_label = os.path.basename(probe_img).split('.')[0]
        # probe_labels.append(id_label)
        probe_labels.append(id_label.split('_0')[0])


poses = list(range(1,13))

similarities = {}

for prob_emb, probe_label in zip(probe_emb, probe_labels):
    current_positives = [f for f in synthetic_content if '_'.join(f.split('_')[0:3]) != probe_label]

    current_positives = random.sample(current_positives, 500)

    similarities[probe_label] = []

    for curr_file in current_positives:
        full_path = os.path.join(afw_directory_synthetic, curr_file)
        img = Image.open(full_path)
        img = lightcnn4_transform(img)[None,:,:,:]
        with torch.no_grad():
            emb = model(img.to(device))
            emb = emb.detach().cpu().numpy()
            probe_emb_np = prob_emb.detach().cpu().numpy()
            similarity = cosine_similarity(probe_emb_np, emb)
            similarity = similarity.squeeze()
            similarities[probe_label].append((curr_file, similarity))
    
all_sims = [sim for file_sims in similarities.values() for _, sim in file_sims]
overall_avg = sum(all_sims) / len(all_sims)

print('Overall Average Similarity: ' + str(overall_avg))
