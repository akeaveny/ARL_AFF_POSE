import numpy as np
import shutil
import glob
import os

import scipy.io
import scipy.misc
from PIL import Image

# =================== new directory ========================
data_path = ''
new_data_path = '/data/Akeaveny/Datasets/part-affordance_combined/real/combined_tools1_'
offset = 0

# =================== directories ========================
### clutter1 = '/data/Akeaveny/Datasets/part-affordance-dataset/part-affordance-clutter/clutter/scene_0*/scene_0*'
BASE = '/data/Akeaveny/Datasets/part-affordance_combined/real/part-affordance-tools/tools/'
images_path1 = BASE + 'bowl_*/bowl_*_*'
images_path2 = BASE + 'cup_*/cup_*_*'
images_path3 = BASE + 'hammer_*/hammer_*_*'
images_path4 = BASE + 'knife_*/knife_*_*'
images_path5 = BASE + 'ladle_*/ladle_*_*'
images_path6 = BASE + 'mallet_*/mallet_*_*'
images_path7 = BASE + 'mug_*/mug_*_*'
images_path8 = BASE + 'pot_*/pot_*_*'
images_path9 = BASE + 'saw_*/saw_*_*'
images_path10 = BASE + 'scissors_*/scissors_*_*'
images_path11 = BASE + 'scoop_*/scoop_*_*'
images_path12 = BASE + 'shears_*/shears_*_*'
images_path13 = BASE + 'shovel_*/shovel_*_*'
images_path14 = BASE + 'spoon_*/spoon_*_*'
images_path15 = BASE + 'tenderizer_*/tenderizer_*_*'
images_path16 = BASE + 'trowel_*/trowel_*_*'
images_path17 = BASE + 'turner_*/turner_*_*'

objects = [images_path1, images_path2, images_path3, images_path4, images_path5,
           images_path6, images_path7, images_path8,images_path9, images_path10,
           images_path11, images_path12, images_path13, images_path14, images_path15,
           images_path16, images_path17]

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