import numpy as np
import shutil
import glob
import os

import scipy.io
import scipy.misc
from PIL import Image

# =================== new directory ========================
data_path = ''
new_data_path = '/data/Akeaveny/Datasets/part-affordance_combined/real/combined_clutter_'
offset = 0

# =================== directories ========================
BASE = '/data/Akeaveny/Datasets/part-affordance_combined/real/part-affordance-clutter/clutter/'
images_path1 = BASE + 'scene_01/scene_*'
images_path2 = BASE + 'scene_02/scene_*'
images_path3 = BASE + 'scene_03/scene_*'

objects = [images_path1,
           images_path2,
           images_path3]

# 2.
scenes = ['']

# 3.
splits = ['']

train_val_split = 0.7
val_test_split = 0.5 # 30% val split are val / test

# 4.
cameras = ['']

# =================== images ext ========================
image_ext1 = '_rgb.jpg'
image_ext2 = '_depth.png'
image_ext3 = '_label.mat'

image_exts = [
            image_ext1,
            image_ext2,
            image_ext3
]

# =================== new directory ========================
offset_train, offset_val, offset_test = 0, 0, 0
train_files_len, val_files_len, test_files_len = 0, 0, 0
for split in splits:
    offset = 0
    for object in objects:
        for scene in scenes:
            for camera in cameras:
                files_offset = 0
                for image_ext in image_exts:
                    file_path = data_path + object + scene + split + camera + image_ext
                    print("File path: ", file_path)
                    files = np.array(sorted(glob.glob(file_path)))
                    print("Loaded files: ", len(files))
                    print("offset: ", offset)

                    ###############
                    # split files
                    ###############
                    np.random.seed(0)
                    total_idx = np.arange(0, len(files), 1)
                    train_idx = np.random.choice(total_idx, size=int(train_val_split * len(total_idx)), replace=False)
                    val_test_idx = np.delete(total_idx, train_idx)

                    train_files = files[train_idx]
                    val_test_files = files[val_test_idx]
                    val_test_idx = np.arange(0, len(val_test_files), 1)

                    val_idx = np.random.choice(val_test_idx, size=int(val_test_split * len(val_test_idx)), replace=False)
                    test_idx = np.delete(val_test_idx, val_idx)
                    val_files = files[val_idx]
                    test_files = files[test_idx]

                    print("Chosen Train Files {}/{}".format(len(train_files), len(files)))
                    print("Chosen Val Files {}/{}".format(len(val_files), len(files)))
                    print("Chosen Test Files {}/{}".format(len(test_files), len(files)))

                    if image_ext == '_rgb.jpg':
                        train_files_len = len(train_files)
                        val_files_len = len(val_files)
                        test_files_len = len(test_files)

                    ###############
                    # train
                    ###############
                    split_folder = 'train/'

                    for idx, file in enumerate(train_files):
                        old_file_name = file
                        folder_to_save = data_path + object + scene + split + camera
                        folder_to_move = new_data_path + split_folder + scene

                        # image_num = offset + idx
                        count = 1000000 + offset_train + idx
                        image_num = str(count)[1:]
                        # print("image_num: ", image_num)

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

                    ###############
                    # val
                    ###############
                    split_folder = 'val/'

                    for idx, file in enumerate(val_files):
                        old_file_name = file
                        folder_to_save = data_path + object + scene + split + camera
                        folder_to_move = new_data_path + split_folder + scene

                        # image_num = offset + idx
                        count = 1000000 + offset_val + idx
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

                    ###############
                    # test
                    ###############
                    split_folder = 'test/'

                    for idx, file in enumerate(test_files):
                        old_file_name = file
                        folder_to_save = data_path + object + scene + split + camera
                        folder_to_move = new_data_path + split_folder + scene

                        # image_num = offset + idx
                        count = 1000000 + offset_test + idx
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

                offset_train += train_files_len
                offset_val += val_files_len
                offset_test += test_files_len