import os
import shutil
from pytorch_image_generation_metrics import (
    get_inception_score_from_directory,
    get_fid_from_directory,
    get_inception_score_and_fid_from_directory)

# case 1
source_path = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/deformable_cycleGAN_w_id_w_pose/results_secondlast_checkpoint_pose_corrected/multipie_pretrained/test_latest/images_root'

source_path = r'/media/oem/storage01/Shakeel/aerial-segmentation/SAM_finetuning/abc/cycleGAN_all/final/calculate_FID/cyclegan_w_id_w_pose_corrected'


poses_all = ['110', '240', '120', '010', '090', '200', '080', '190', '130', '041', '140', '050', '051']
pose_names = ['-90', '+90', '-75', '+75', '-60', '+60', '-45', '+45', '-30', '+30', '-15', '+15', '0']
img_counts_lst = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
emot_count_lst = ['01', '02']
del poses_all[-1]
del pose_names[-1]

image_ids = list(range(151,251))
del image_ids[image_ids.index(213)] # remove 218 as it is not present in the dataset

target_root = 'temp_is'
if not os.path.exists(target_root):
    os.makedirs(target_root)

for curr_pose, curr_pose_name in zip(poses_all, pose_names):
    curr_pose_dir = os.path.join(target_root, curr_pose_name)
    if not os.path.exists(curr_pose_dir):
        os.makedirs(curr_pose_dir)
    lpips_scores = []
    for img_id in image_ids:
        for emot in emot_count_lst:
            for img_count in img_counts_lst:
                image_name_predicted = f"{img_id}_01_{emot}_{curr_pose}_{img_count}_crop_128_fake.png"
                # image_name_groundtruth = f"{img_id}_01_{emot}_051_{img_count}_crop_128.png"

                predicted_path = os.path.join(source_path, image_name_predicted)
                target_path = os.path.join(curr_pose_dir, image_name_predicted)
                shutil.copyfile(predicted_path, target_path)


    IS, IS_std = get_inception_score_from_directory(
        curr_pose_dir)

    print(f'Inception Score | pose {curr_pose_name}: {IS} ± {IS_std}')

    # shutil.rmtree(curr_pose_dir)