import imageio
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

from lightcnn.lightcnn import LightCNN_29Layers_v2
from lightcnn.light_cnn_v4 import LightCNN_V4

import os
import  torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from torch.utils.data import DataLoader


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


# data_path = r'multi_PIE_crop_128_test_probe_and_gallery'
# data_path = r'deformable_w_id_w_pose_lastest_ultrapromax'
# data_path = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cycleGAN_w_id_w_pose/secondlast_checkpoint_pose_corrected'
data_path = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cyclegan_w_id_w_pose_middledeform/results/multipie_pretrained/test_latest/secondlast_checkpoint_middledeform'
data_path = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cycleGAN_w_id_w_pose/results_afw_second_last/multipie_pretrained/test_latest/images_structure'
data_path = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/cyclegan_w_id_w_pose_loss_corrected/results_afw/multipie_pretrained/test_latest/images_structure'
data_path = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cyclegan_w_id_w_pose_middledeform/results/multipie_pretrained/test_latest/images_structure'
data_path = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cyclegan_id_pose_resnetdeform/results/multipie_pretrained/test_latest/images_structure'

probe_list = [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]


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
        probe_labels.append(id_label)

poses = {
    '90': ('110', '240'),
    '75': ('120', '010'),
    '60': ('090', '200'),
    '45': ('080', '190'),
    '30': ('130', '041'),
    '15': ('140', '050')
}

poses = list(range(1,13))

threshold = 0.5

for i in poses:
    pose_code = i

    dataset = ImageDataset(data_path, lightcnn4_transform, pose_code)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=12)

    predictions_all = []
    labels_all = []

    for j in range(len(probe_emb)):
        curr_probe_emb = probe_emb[j]
        curr_probe_emb = curr_probe_emb.reshape(1, -1)
        curr_probe_emb = curr_probe_emb.detach().cpu().numpy()
        curr_probe_label = probe_labels[j]
        for step, (imgs, labels) in enumerate(dataloader):

            imgs = imgs.to(device)
            labels = list(labels)

            with torch.no_grad():
                embeddings = model(imgs)

                _probe_labels = [curr_probe_label] * imgs.size()[0]

                embeddings = embeddings.detach().cpu().numpy()

                similarities = cosine_similarity(curr_probe_emb, embeddings)
                similarities = similarities.squeeze()
                predictions = [1 if similarity >= threshold else 0 for similarity in similarities]

                ground_truth  = [1 if a == b else 0 for a, b in zip(_probe_labels, labels)]

                predictions_all.extend(predictions)
                labels_all.extend(ground_truth)
    
    acc_result = metrics.accuracy_score(predictions_all, labels_all)

    print('Accuracy of Pose ' + str(pose_code) + ' :' + str(acc_result))
    
