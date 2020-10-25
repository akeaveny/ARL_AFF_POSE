import numpy as np
import shutil
import glob
import os

import scipy.io
import scipy.misc
from PIL import Image

import cv2

import matplotlib.pyplot as plt

# =================== new directory ========================
# 0.
data_path = '/data/Akeaveny/Datasets/part-affordance_combined/real/part-affordance-tools/tools/'
new_data_path = '/data/Akeaveny/Datasets/part-affordance_combined/real/combined_tools_test_scissors/'

# =================== load from ========================
# 1.
np.random.seed(0)
num_test = int(10)
test_idx = np.random.choice(np.arange(0, 200, 1), size=int(num_test), replace=False)
print("Chosen Files \n", test_idx)

objects = [
        'scissors_01/', 'scissors_02/', 'scissors_03/', 'scissors_04/', 'scissors_05/',
        'scissors_06/', 'scissors_07/', 'scissors_08/']

# 2.
scenes = [
            '']

# 3.
splits = [
          '']

# 4.
cameras = [
            '']

# =================== images ext ========================
image_ext10 = '_rgb.jpg'
image_ext20 = '_depth.png'
image_ext30 = '_label.mat'
image_exts1 = [
    image_ext10,
    image_ext20,
    image_ext30,
]

# =================== new directory ========================
for split in splits:
    offset = 0
    for object in objects:
        for scene in scenes:
            for camera in cameras:
                files_offset = 0
                for image_ext in image_exts1:
                    file_path = data_path + object + scene + split + camera + '*' + image_ext
                    print("File path: ", file_path)
                    files = np.array(sorted(glob.glob(file_path)))
                    print("Loaded files: ", len(files))
                    print("offset: ", offset)

                    files = files[test_idx]

                    for idx, file in enumerate(files):
                        old_file_name = file
                        folder_to_save = data_path + object + scene + split + camera
                        folder_to_move = new_data_path + split + '/' + scene

                        # image_num = offset + idx
                        count = 1000000 + offset + idx
                        image_num = str(count)[1:]

                        if image_ext == '_rgb.jpg':
                            new_file_name = folder_to_save + np.str(image_num) + '_rgb.jpg'
                            move_file_name = folder_to_move + np.str(image_num) + '_rgb.jpg'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '_depth.png':
                            new_file_name = folder_to_save + np.str(image_num) + '_depth.png'
                            move_file_name = folder_to_move + np.str(image_num) + '_depth.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '_label.mat':
                            img = np.array(scipy.io.loadmat(file)['gt_label'], dtype=np.uint8).reshape(480, 640)
                            new_file_name = folder_to_save + np.str(image_num) + '_label.png'
                            move_file_name = folder_to_move + np.str(image_num) + '_label.png'
                            im = Image.fromarray(np.uint8(img))
                            im.save(move_file_name)
                        else:
                            pass

                        # print("Old File: ", old_file_name)
                        # print("New File: ", new_file_name)e)

                offset += len(files)