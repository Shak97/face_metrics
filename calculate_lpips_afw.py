# 2) save as lpips_score.py and run: python lpips_score.py --img0 path/to/a.jpg --img1 path/to/b.jpg --net alex
import argparse, torch
import lpips
from PIL import Image
import torchvision.transforms as T
import os

def load_for_lpips(path):
    img = Image.open(path).convert("RGB")
    # LPIPS expects [-1, 1] normalized tensors, shape (N,3,H,W)
    to_tensor = T.Compose([
        T.ToTensor(),                                 # [0,1]
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  # -> [-1,1]
        T.Resize((256,256))
    ])
    t = to_tensor(img).unsqueeze(0)
    return t


loss_fn = lpips.LPIPS('alex')   # alex/vgg/squeeze
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# groundtruth_dir = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cycleGAN_w_id/data sets/multipie/testB'
# predicted_dir = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cycleGAN_w_id/results/multipie_pretrained/test_latest/images'
# predicted_dir = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cycleGAN_w_id_w_pose/results_secondlast_checkpoint_pose_corrected/multipie_pretrained/test_latest/images'
# predicted_dir = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cyclegan_w_id_w_pose_middledeform/results_secondlast_checkpoint_middledeform/multipie_pretrained/test_latest/images'
# predicted_dir = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cyclegan_wo_id_w_pose_corrected/results_checkpoint/multipie_pretrained/test_latest/images'
# predicted_dir = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/cyclegan_w_id_w_pose_loss_corrected/results_checkpoint/multipie_pretrained/test_latest/images'
# predicted_dir = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_w_id_w_pose_corrected_first_layers/testimages_fake/'

# afw dataset paths
groundtruth_dir = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cycleGAN_w_id_w_pose/datasets/multipie/AFW_cropped'
predicted_dir = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cycleGAN_w_id_w_pose/results_afw_second_last/multipie_pretrained/test_latest/images_root'

predicted_dir = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/ablation_hyperparam/case4/results_afw/multipie_pretrained/test_latest/images_root'
predicted_dir = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/cyclegan_w_id_w_pose_loss_corrected/results_afw/multipie_pretrained/test_latest/images_root'

# poses_all = ['110', '240', '120', '010', '090', '200', '080', '190', '130', '041', '140', '050', '051']
# pose_names = ['-90', '+90', '-75', '+75', '-60', '+60', '-45', '+45', '-30', '+30', '-15', '+15', '0']
# img_counts_lst = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
# emot_count_lst = ['01', '02']
# del poses_all[-1]
# del pose_names[-1]

# image_ids = list(range(151,251))
# del image_ids[image_ids.index(213)]  # remove 218 as it is not present in the dataset

# for curr_pose, curr_pose_name in zip(poses_all, pose_names):
#     lpips_scores = []
#     for img_id in image_ids:
#         for emot in emot_count_lst:
#             for img_count in img_counts_lst:
#                 image_name_predicted = f"{img_id}_01_{emot}_{curr_pose}_{img_count}_crop_128_fake.png"
#                 image_name_groundtruth = f"{img_id}_01_{emot}_051_{img_count}_crop_128.png"

#                 predicted_path = os.path.join(predicted_dir, image_name_predicted)
#                 groundtruth_path = os.path.join(groundtruth_dir, image_name_groundtruth)

#                 im0 = load_for_lpips(predicted_path)
#                 im1 = load_for_lpips(groundtruth_path)
#                 # if torch.cuda.is_available():
#                 im0 = im0.cuda(); im1 = im1.cuda()

#                 with torch.no_grad():
#                     d = loss_fn(im0, im1).item()
#                     lpips_scores.append(d)

#     avg_lpips = sum(lpips_scores) / len(lpips_scores)
#     pose_index = poses_all.index(curr_pose)
#     print(f"Average LPIPS ({curr_pose_name}) : {avg_lpips:.6f}")

lpips_scores = []
content = os.listdir(predicted_dir)
for file_name in content:
    if file_name.split('.')[0].split('_')[-1] == 'real':
        continue
    predicted_path = os.path.join(predicted_dir, file_name)
    filename_content = file_name.split('_')
    del filename_content[-1]
    new_file_name = '_'.join(filename_content) + '_0.jpg'
    groundtruth_path = os.path.join(groundtruth_dir, new_file_name)

    # if not os.path.exists(groundtruth_path):
    #     print(f'skipping {groundtruth_path}')
    #     continue

    im0 = load_for_lpips(predicted_path)
    im1 = load_for_lpips(groundtruth_path)
    # if torch.cuda.is_available():
    im0 = im0.cuda(); im1 = im1.cuda()

    with torch.no_grad():
        d = loss_fn(im0, im1).item()
        lpips_scores.append(d)

avg_lpips = sum(lpips_scores) / len(lpips_scores)
print(f"Average LPIPS: {avg_lpips:.6f}")

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--img0", required=True)
#     p.add_argument("--img1", required=True)
#     p.add_argument("--net", default="alex", choices=["alex","vgg","squeeze"])
#     p.add_argument("--use_gpu", action="store_true")
#     args = p.parse_args()

#     loss_fn = lpips.LPIPS(net=args.net)   # alex/vgg/squeeze
#     if args.use_gpu and torch.cuda.is_available():
#         loss_fn = loss_fn.cuda()

#     im0 = load_for_lpips(args.img0)
#     im1 = load_for_lpips(args.img1)
#     if args.use_gpu and torch.cuda.is_available():
#         im0 = im0.cuda(); im1 = im1.cuda()

#     with torch.no_grad():
#         d = loss_fn(im0, im1).item()

#     print(f"LPIPS ({args.net}) between\n  {args.img0}\n  {args.img1}\n= {d:.6f}")

# if __name__ == "__main__":
#     main()
