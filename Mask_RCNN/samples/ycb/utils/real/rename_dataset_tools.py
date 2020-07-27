import numpy as np
import glob
import shutil
import os

import scipy.io as sio

####################
# init
####################
# dir
data_path = '/data/Akeaveny/Datasets/YCB_Video_Dataset/'
new_data_path = '/data/Akeaveny/Datasets/YCB_Video_Dataset/data_combined/'

####################
# image ext
####################

splits = [
    'val/',
    'train/'
]

folder_paths = [
    '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/ycb/dataset_config/test_data_list.txt',
    '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/ycb/dataset_config/train_data_list.txt'
]

image_ext10 = '-box.txt'
image_ext20 = '-color.png'
image_ext30 = '-depth.png'
image_ext40 = '-label.png'
image_ext50 = '-meta.mat'
image_exts1 = [
    image_ext10,
    image_ext20,
    image_ext30,
    image_ext40,
    image_ext50
]

# =================== new directory ========================
for split_idx, split in enumerate(splits):
    offset = 0
    files_offset = 0

    folders = folder_paths[split_idx]
    image_paths = np.loadtxt('{}'.format(folders), dtype=np.str)
    print("Folders path: {}".format(folders))
    print("Loaded images: ", len(image_paths))
    print("offset: ", offset)

    for image_idx, image_path in enumerate(image_paths[0:1]):
        for image_ext in image_exts1:

            image_path_ = data_path + image_path + image_ext
            image = sorted(glob.glob(image_path_))

            old_file_name = image_path_
            folder_to_move = new_data_path + split

            # image_num = offset + idx
            count = 1000000 + offset + image_idx
            image_num = str(count)[1:]

            if image_ext == '-box.txt':
                move_file_name = folder_to_move + np.str(image_num) + '_box.txt'
                print("old_file_name: ", old_file_name)
                print("move_file_name: ", move_file_name)

            elif image_ext == '-color.png':
                move_file_name = folder_to_move + np.str(image_num) + '_rgb.png'
                print("old_file_name: ", old_file_name)
                print("move_file_name: ", move_file_name)

            elif image_ext == '-depth.png':
                move_file_name = folder_to_move + np.str(image_num) + '_depth.png'
                print("old_file_name: ", old_file_name)
                print("move_file_name: ", move_file_name)

            elif image_ext == '-label.png':
                move_file_name = folder_to_move + np.str(image_num) + '_label.png'
                print("old_file_name: ", old_file_name)
                print("move_file_name: ", move_file_name)

            elif image_ext == '-meta.mat':
                move_file_name = folder_to_move + np.str(image_num) + '_meta.mat'
                print("old_file_name: ", old_file_name)
                print("move_file_name: ", move_file_name)

            else:
                print('\n***************** Unknown File *****************')
                print(image_path_)
                exit(1)

            shutil.copyfile(old_file_name, move_file_name)

        # offset += len(images_paths)