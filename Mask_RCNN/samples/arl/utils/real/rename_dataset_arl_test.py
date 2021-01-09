import numpy as np
import shutil
import glob
import os

from yaml_to_mat import yaml_to_mat
import scipy.io as sio

############################
# TOOLS
############################
# data_path = '/data/Akeaveny/Datasets/arl_dataset/test/tools/arl_test_tools_1/'
# new_data_path = '/data/Akeaveny/Datasets/arl_dataset/combined_test_tools_3_'
#
############################
# subfolder = 'images/'
# objects = [
#     'test_single_garden_shovel_1/' + subfolder,
#     'test_single_mallet_1/' + subfolder,
#     'test_single_screwdriver_1/' + subfolder,
#     'test_single_spatula_1/' + subfolder,
#     'test_single_wooden_spoon_1/' + subfolder,
#     ]

# ############################
# # CLUTTER
# ############################
data_path = '/data/Akeaveny/Datasets/arl_dataset/test/clutter/arl_test_clutter_1/'
new_data_path = '/data/Akeaveny/Datasets/arl_dataset/combined_test_clutter_3_'

############################
subfolder = 'images/'
objects = [
    'test_clutter_scene_1/' + subfolder,
    ]

# 2.
scenes =  ['']

# 3.
splits = ['']


# 4.
cameras = ['']

#
train_val_split = 0.7
val_test_split = 0.65 # 30% val split are val / test

# =================== images ext ========================
image_ext10 = '_rgb.png'
image_ext20 = '_depth.png'
image_ext30 = '_labels.png'
image_ext40 = '_poses.yaml'
image_exts = [
    image_ext10,
    image_ext20,
    image_ext30,
    image_ext40,
]

# =================== new directory ========================
offset_train, offset_val, offset_test = 0, 0, 0
train_files_len, val_files_len, test_files_len = 0, 0, 0
for split in splits:
    for object in objects:
        for scene in scenes:
            for camera in cameras:
                files_offset = 0
                for image_ext in image_exts:
                    file_path = data_path + object + scene + split + camera + '*' + image_ext
                    print("File path: ", file_path)
                    files = np.array(sorted(glob.glob(file_path)))
                    print("offset: ", offset_train, offset_val, offset_test)
                    print("Loaded files: ", len(files))

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

                    if image_ext == '_poses.yaml':
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

                        if image_ext == '_labels.png':
                            # print("BOOM LABEL")
                            move_file_name = folder_to_move + np.str(image_num) + '_label.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '_depth.png':
                            # print("BOOM DEPTH")
                            move_file_name = folder_to_move + np.str(image_num) + '_depth.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '_rgb.png':
                            # print("BOOM RGB")
                            move_file_name = folder_to_move + np.str(image_num) + '_rgb.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == "_poses.yaml":

                            mat = yaml_to_mat(file)

                            move_mat_name = folder_to_move + np.str(image_num) + '-meta.mat'
                            sio.savemat(move_mat_name, mat)

                        else:
                            print("*** IMAGE EXT {} DOESN'T EXIST ***".format(image_ext))
                            exit(1)

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

                        if image_ext == '_labels.png':
                            # print("BOOM LABEL")
                            move_file_name = folder_to_move + np.str(image_num) + '_label.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '_depth.png':
                            # print("BOOM DEPTH")
                            move_file_name = folder_to_move + np.str(image_num) + '_depth.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '_rgb.png':
                            # print("BOOM RGB")
                            move_file_name = folder_to_move + np.str(image_num) + '_rgb.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == "_poses.yaml":

                            mat = yaml_to_mat(file)

                            move_mat_name = folder_to_move + np.str(image_num) + '-meta.mat'
                            sio.savemat(move_mat_name, mat)

                        else:
                            print("*** IMAGE EXT {} DOESN'T EXIST ***".format(image_ext))
                            exit(1)

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

                        if image_ext == '_labels.png':
                            # print("BOOM LABEL")
                            move_file_name = folder_to_move + np.str(image_num) + '_label.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '_depth.png':
                            # print("BOOM DEPTH")
                            move_file_name = folder_to_move + np.str(image_num) + '_depth.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '_rgb.png':
                            # print("BOOM RGB")
                            move_file_name = folder_to_move + np.str(image_num) + '_rgb.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == "_poses.yaml":

                            mat = yaml_to_mat(file)

                            move_mat_name = folder_to_move + np.str(image_num) + '-meta.mat'
                            sio.savemat(move_mat_name, mat)

                        else:
                            print("*** IMAGE EXT {} DOESN'T EXIST ***".format(image_ext))
                            exit(1)

                ###############
                ###############
                offset_train += train_files_len
                offset_val += val_files_len
                offset_test += test_files_len