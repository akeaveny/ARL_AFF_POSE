import numpy as np
import shutil
import glob
import os

from json_to_mat import json_to_mat
import scipy.io as sio

# =================== new directory ========================
# 0.
data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/umd_affordance1/'
new_data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/combined_tools5_'

# =================== load from ========================

objects = [
    'bowl_01/',  'bowl_02/',  'bowl_03/',  'bowl_04/',  'bowl_05/',
    'bowl_06/',  'bowl_07/',

    'cup_01/', 'cup_02/', 'cup_03/', 'cup_04/', 'cup_05/',
    'cup_06/',

    'hammer_01/', 'hammer_02/', 'hammer_03/', 'hammer_04/',

    'knife_01/', 'knife_02/', 'knife_03/', 'knife_04/', 'knife_05/',
    'knife_06/', 'knife_07/', 'knife_08/', 'knife_09/', 'knife_10/',
    'knife_11/', 'knife_12/',

    'ladle_01/', 'ladle_02/', 'ladle_03/',

    'mallet_01/', 'mallet_02/', 'mallet_03/', 'mallet_04/',

    'mug_01/', 'mug_02/', 'mug_03/', 'mug_04/', 'mug_05/',
    'mug_06/', 'mug_07/', 'mug_08/', 'mug_09/', 'mug_10/',
    'mug_11/', 'mug_12/', 'mug_13/', 'mug_14/', 'mug_15/',
    'mug_16/', 'mug_17/', 'mug_18/', 'mug_19/', 'mug_20/'

    'pot_01/', 'pot_02/',

    'saw_01/', 'saw_02/', 'saw_03/',

    'scissors_01/', 'scissors_02/', 'scissors_03/', 'scissors_04/', 'scissors_05/',
    'scissors_06/', 'scissors_07/', 'scissors_08/',

    'scoop_01/', 'scoop_02/',

    'shears_01/', 'shears_02/',

    'shovel_01/', 'shovel_02/',

    'spoon_01/', 'spoon_02/', 'spoon_03/', 'spoon_04/', 'spoon_05/',
    'spoon_06/', 'spoon_07/', 'spoon_08/', 'spoon_09/', 'spoon_10/',

    'tenderizer_01/',

    'trowel_01/', 'trowel_02/', 'trowel_03/',

    'turner_01/', 'turner_02/', 'turner_03/', 'turner_04/', 'turner_05/',
    'turner_06/', 'turner_07/'
    ]

# 2.
scenes = [
        'bench/',
        'floor/',
        'turn_table/',
        'dr/'
          ]

# 3.
splits = [
          'train/',
          ]

train_val_split = 0.7
val_test_split = 0.5 # 30% val split are val / test

# 4.
cameras = [
    'Kinect/',
    'Xtion/',
    'ZED/'
]

# =================== images ext ========================
image_ext10 = '.json'
image_ext20 = '.cs.png'
image_ext30 = '.depth.mm.16.png'
image_ext40 = '.depth.png'
image_ext50 = '.png'
image_exts = [
    image_ext20,
    image_ext40,
    image_ext50,
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
                    file_path = data_path + object + scene + split + camera + '??????' + image_ext
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

                    if image_ext == '.png':
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

                        if image_ext == '.cs.png':
                            # print("BOOM LABEL")
                            move_file_name = folder_to_move + np.str(image_num) + '_label.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '.depth.png':
                            # print("BOOM DEPTH")
                            move_file_name = folder_to_move + np.str(image_num) + '_depth.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '.png':
                            # print("BOOM RGB")
                            move_file_name = folder_to_move + np.str(image_num) + '_rgb.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == ".json":
                            move_file_name = folder_to_move + np.str(image_num) + '.json'

                            json_num = file.split("/")[-1].split(".json")[0]
                            json_file = folder_to_save + np.str(json_num) + '.json'

                            camera_settings = camera.split("/")[0]
                            mat = json_to_mat(json_file, camera_settings)

                            move_mat_name = folder_to_move + np.str(image_num) + '-meta.mat'
                            sio.savemat(move_mat_name, mat)

                        else:
                            print("*** IMAGE EXT DOESN'T EXIST ***")
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

                        if image_ext == '.cs.png':
                            move_file_name = folder_to_move + np.str(image_num) + '_label.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '.depth.png':
                            move_file_name = folder_to_move + np.str(image_num) + '_depth.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '.png':
                            move_file_name = folder_to_move + np.str(image_num) + '_rgb.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == ".json":
                            move_file_name = folder_to_move + np.str(image_num) + '.json'

                            json_num = file.split("/")[-1].split(".json")[0]
                            json_file = folder_to_save + np.str(json_num) + '.json'

                            camera_settings = camera.split("/")[0]
                            mat = json_to_mat(json_file, camera_settings)

                            move_mat_name = folder_to_move + np.str(image_num) + '-meta.mat'
                            sio.savemat(move_mat_name, mat)

                        else:
                            print("*** IMAGE EXT DOESN'T EXIST ***")
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

                        if image_ext == '.cs.png':
                            move_file_name = folder_to_move + np.str(image_num) + '_label.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '.depth.png':
                            move_file_name = folder_to_move + np.str(image_num) + '_depth.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '.png':
                            move_file_name = folder_to_move + np.str(image_num) + '_rgb.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == ".json":
                            move_file_name = folder_to_move + np.str(image_num) + '.json'

                            json_num = file.split("/")[-1].split(".json")[0]
                            json_file = folder_to_save + np.str(json_num) + '.json'

                            camera_settings = camera.split("/")[0]
                            mat = json_to_mat(json_file, camera_settings)

                            move_mat_name = folder_to_move + np.str(image_num) + '-meta.mat'
                            sio.savemat(move_mat_name, mat)

                        else:
                            print("*** IMAGE EXT DOESN'T EXIST ***")
                            exit(1)

                ###############
                ###############
                offset_train += train_files_len
                offset_val += val_files_len
                offset_test += test_files_len

