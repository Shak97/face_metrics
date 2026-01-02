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

# from mobilefacenet_lfw_99 import MobileFaceNet
from mobilefacenet_new import MobileFacenet


from dataset_lightcnnv4 import ImageDataset

# images_path = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cycleGAN_w_id_w_pose/lfw_fake_sub/'
images_path = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/datasets/lfw_cut/'
# images_path = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cycleGAN_w_id_w_pose/lfw_fake__latest_sub'
pairsfile = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cycleGAN_w_id_w_pose/pairsDevTest.txt'

with open(pairsfile, 'r') as f:
    paircontent = f.read()

pair_content = paircontent.split('\n')
del pair_content[-1]


device = torch.device(3)

model = LightCNN_V4(None)
# model = MobileFacenet()
state_dict = torch.load(r'lightcnn/LightCNN-V4_checkpoint.pth.tar', map_location=device)['state_dict']
# state_dict = torch.load(r'068.ckpt', map_location=device)['net_state_dict']

# model = torch.jit.load(r'mobilefacenet_scripted (2).pt')

print(model)

model.load_state_dict(state_dict, strict = True)
model = model.to(device)

# lightcnn4_transform = transforms.Compose([
#     transforms.Resize((112,112)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

lightcnn4_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 255.0 - 127.5) / 128.0)
])

same = 3
diff  = 4

threshold = 0.5

pred_labels = []
real_labels = []

count = 0

with torch.no_grad():

    for pair in pair_content:
        pc = pair.split('\t')

        if pc is not None and len(pc) > 1:
            if len(pc) == 3:
                label = 1
                dirname = pc[0]
                img1 = pc[1]
                img2 = pc[2]

                ppp = os.path.join(images_path, dirname)

                if os.path.exists(ppp):

                    allimages = os.listdir(os.path.join(images_path, dirname))

                    im1 = Image.open(os.path.join(images_path, dirname,allimages[int(img1) - 1])).convert('RGB')
                    try:
                        im2 = Image.open(os.path.join(images_path, dirname, allimages[int(img2) - 1])).convert('RGB')
                    except:
                        im2 = Image.open(os.path.join(images_path, dirname, allimages[int(img1) - 1])).convert('RGB')


                    im1 = lightcnn4_transform(im1)[None,:,:,:]
                    im2 = lightcnn4_transform(im2)[None,:,:,:]

                    pred1 = model(im1.to(device))
                    pred2 = model(im2.to(device))

                    emb1 = pred1.detach().cpu().numpy()
                    emb2 = pred1.detach().cpu().numpy()

                    similarities = cosine_similarity(emb1, emb2)
                    similarities = similarities.squeeze()

                    plabel = 1 if similarities >= threshold else 0
                    pred_labels.append(float(similarities))

                    real_labels.append(label)
                else:
                    count +=1



            else:
                label = 0

                dirname1 = pc[0]
                img1 = pc[1]
                dirname2 = pc[2]
                img2 = pc[3]

                bbb = os.path.join(images_path, dirname1)
                if not os.path.exists(bbb):
                    count+=1
                    continue

                allimages1 = os.listdir(bbb)

                ccc = os.path.join(images_path, dirname2)

                if not os.path.exists(ccc):
                    count+=1
                    continue
                allimages2 = os.listdir(os.path.join(images_path, dirname2))

                try:
                    im1 = Image.open(os.path.join(images_path, dirname1, allimages1[int(img1) -1])).convert('RGB')
                except:
                    im1 = Image.open(os.path.join(images_path, dirname1, allimages1[0])).convert('RGB')

                im2 = Image.open(os.path.join(images_path, dirname2, allimages2[int(img2) -1])).convert('RGB')

                im1 = lightcnn4_transform(im1)[None,:,:,:]
                im2 = lightcnn4_transform(im2)[None,:,:,:]


                pred1 = model(im1.to(device))
                pred2 = model(im2.to(device))

                emb1 = pred1.detach().cpu().numpy()
                emb2 = pred1.detach().cpu().numpy()

                similarities = cosine_similarity(emb1, emb2)
                similarities = similarities.squeeze()
                print(similarities)
                plabel = 1 if similarities >= threshold else 0
                # pred_labels.append(plabel)
                pred_labels.append(float(similarities))


                real_labels.append(label)

print(count)
acc_result = metrics.accuracy_score(pred_labels, real_labels)
print(acc_result)




        


# threshold = 0.5

# for i in poses.keys():
#     pose_code = i

#     dataset = ImageDataset(data_path, lightcnn4_transform, pose_code)
#     dataloader = DataLoader(dataset, batch_size=64, num_workers=12)

#     predictions_all = []
#     labels_all = []

#     for j in range(len(probe_emb)):
#         curr_probe_emb = probe_emb[j]
#         curr_probe_emb = curr_probe_emb.reshape(1, -1)
#         curr_probe_emb = curr_probe_emb.detach().cpu().numpy()
#         curr_probe_label = probe_labels[j]
#         for step, (imgs, labels) in enumerate(dataloader):

#             imgs = imgs.to(device)
#             labels = list(labels)

#             with torch.no_grad():
#                 embeddings = model(imgs)

#                 _probe_labels = [curr_probe_label] * imgs.size()[0]

#                 embeddings = embeddings.detach().cpu().numpy()

#                 similarities = cosine_similarity(curr_probe_emb, embeddings)
#                 similarities = similarities.squeeze()
#                 predictions = [1 if similarity >= threshold else 0 for similarity in similarities]

#                 ground_truth  = [1 if a == b else 0 for a, b in zip(_probe_labels, labels)]

#                 predictions_all.extend(predictions)
#                 labels_all.extend(ground_truth)
    
#     acc_result = metrics.accuracy_score(predictions_all, labels_all)

#     print('Accuracy of Pose ' + str(pose_code) + ' :' + str(acc_result))
    
