import numpy as np
import shutil
import glob
import os

import scipy.io
import scipy.misc
from PIL import Image

# =================== new directory ========================
data_path = '/data/Akeaveny/Datasets/part-affordance_combined/real/combined_tools1_'
new_data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/finetune/umd1_'
offset = 0

# =================== directories ========================
objects = ['']

# 2.
scenes = ['']

# 3.
splits = ['train/',
          'val/']

# 4.
cameras = ['']

# =================== images ext ========================
image_ext1 = '_rgb.jpg'
image_ext2 = '_depth.png'
image_ext3 = '_label.png'

image_exts = [
            image_ext1,
            image_ext2,
            image_ext3
]

NUM_TRAIN_FINETUNING = 500
NUM_VAL_FINETUNING = 125

# =================== new directory ========================
offset_train, offset_val = 0, 0
train_files_len, val_files_len = 0, 0
for split in splits:
    offset = 0
    for object in objects:
        for scene in scenes:
            for camera in cameras:
                files_offset = 0
                for image_ext in image_exts:
                    file_path = data_path + object + scene + split + camera + '*' + image_ext
                    print("File path: ", file_path)
                    files = np.array(sorted(glob.glob(file_path)))
                    print("Loaded files: ", len(files))
                    print("offset: ", offset_train, offset_val)

                    if split == 'train/':
                        NUM_FINETUNING = NUM_TRAIN_FINETUNING
                    else:
                        NUM_FINETUNING = NUM_VAL_FINETUNING

                    ###############
                    # split files
                    ###############
                    np.random.seed(0)
                    total_idx = np.arange(0, len(files), 1)
                    split_idx = np.random.choice(total_idx, size=int(NUM_FINETUNING), replace=False)
                    split_files = files[split_idx]

                    print("Chosen Files {}/{}".format(len(split_files), len(files)))

                    ###############
                    ###############

                    for idx, file in enumerate(split_files):
                        old_file_name = file
                        folder_to_save = data_path + object + scene + split + camera
                        folder_to_move = new_data_path + split + scene

                        count = 1000000 + idx
                        image_num = str(count)[1:]

                        if image_ext == '_rgb.jpg':
                            new_file_name = folder_to_save + np.str(image_num) + '_rgb.jpg'
                            move_file_name = folder_to_move + np.str(image_num) + '_rgb.jpg'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '_depth.png':
                            new_file_name = folder_to_save + np.str(image_num) + '_depth.png'
                            move_file_name = folder_to_move + np.str(image_num) + '_depth.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '_label.png':
                            new_file_name = folder_to_save + np.str(image_num) + '_label.png'
                            move_file_name = folder_to_move + np.str(image_num) + '_label.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        else:
                            pass