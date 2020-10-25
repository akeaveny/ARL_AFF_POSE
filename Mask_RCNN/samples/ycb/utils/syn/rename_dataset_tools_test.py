import numpy as np
import shutil
import glob
import os

from json_to_mat import json_to_mat
import scipy.io as sio

####################
# init
####################
# dir
data_path = '/data/Akeaveny/Datasets/ycb_syn/ycb_affordance/'
new_data_path = '/data/Akeaveny/Datasets/ycb_syn/combined_tools3_'

####################
# scenes
####################
# 1.
folder_to_object = 'ycb_affordance/'

objects = [
        '002_master_chef_can/',
        '003_cracker_box/',
        '004_sugar_box/',
        '005_tomato_soup_can/',
        '006_mustard_bottle/',
        '007_tuna_fish_can/',
        '008_pudding_box/',
        '009_gelatin_box/',
        '010_potted_meat_can/',
        '011_banana/',
        '019_pitcher_base/',
        '021_bleach_cleanser/',
        '024_bowl/',
        '025_mug/',
        '035_power_drill/',
        '036_wood_block/',
        '037_scissors/',
        '040_large_marker/',
        '051_large_clamp/',
        '052_extra_large_clamp/',
        '061_foam_brick/',
            ]

# 2.
scenes = [
        'bench/', 'floor/', 'turn_table/',
        'dr/'
          ]

# 3.
splits = [
          'train/',
          ]

train_val_split = 0.8
val_test_split = 0.5 # 30% val split are val / test

# 4.
cameras = [
    'Kinect/',
    'Xtion/',
    'ZED/'
]

####################
# image ext
####################
image_ext10 = '.json'
image_ext20 = '.cs.png'
image_ext30 = '.depth.cm.8.png'
image_ext40 = '.depth.png'
image_ext50 = '.png'
image_exts1 = [
    image_ext10,
    image_ext20,
    image_ext40,
    image_ext50
]

####################
# new dir
####################
offset_train, offset_val, offset_test = 0, 0, 0
train_files_len, val_files_len, test_files_len = 0, 0, 0
for split in splits:
    for object in objects:
        for scene in scenes:
            for camera in cameras:
                files_offset = 0
                for image_ext in image_exts1:
                    file_path = data_path + object + scene + split + camera + '??????' + image_ext
                    print("\nFile path: ", file_path)
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

                    val_idx = np.random.choice(val_test_idx, size=int(val_test_split * len(val_test_idx)),replace=False)
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

                        if image_ext == ".json":
                            move_file_name = folder_to_move + np.str(image_num) + '.json'

                            json_num = file.split("/")[-1].split(".json")[0]
                            json_file = folder_to_save + np.str(json_num) + '.json'

                            # print("og: ", file)
                            # print("json_file: ", json_file)

                            camera_settings = camera.split("/")[0]
                            mat = json_to_mat(json_file, camera_settings)

                            move_mat_name = folder_to_move + np.str(image_num) + '-meta.mat'
                            sio.savemat(move_mat_name, mat)

                        elif image_ext == ".png":
                            move_file_name = folder_to_move + np.str(image_num) + '_rgb.png'

                        elif image_ext == ".depth.cm.8.png" or image_ext == ".depth.png":
                            move_file_name = folder_to_move + np.str(image_num) + '_depth.png'

                        elif image_ext == ".cs.png":
                            move_file_name = folder_to_move + np.str(image_num) + '_label.png'

                        else:
                            pass

                        shutil.copyfile(old_file_name, move_file_name)

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

                        if image_ext == ".json":
                            move_file_name = folder_to_move + np.str(image_num) + '.json'

                            json_num = file.split("/")[-1].split(".json")[0]
                            json_file = folder_to_save + np.str(json_num) + '.json'

                            # print("og: ", file)
                            # print("json_file: ", json_file)

                            camera_settings = camera.split("/")[0]
                            mat = json_to_mat(json_file, camera_settings)

                            move_mat_name = folder_to_move + np.str(image_num) + '-meta.mat'
                            sio.savemat(move_mat_name, mat)

                        elif image_ext == ".png":
                            move_file_name = folder_to_move + np.str(image_num) + '_rgb.png'

                        elif image_ext == ".depth.cm.8.png" or image_ext == ".depth.png":
                            move_file_name = folder_to_move + np.str(image_num) + '_depth.png'

                        elif image_ext == ".cs.png":
                            move_file_name = folder_to_move + np.str(image_num) + '_label.png'

                        else:
                            pass

                        shutil.copyfile(old_file_name, move_file_name)

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

                        if image_ext == ".json":
                            move_file_name = folder_to_move + np.str(image_num) + '.json'

                            json_num = file.split("/")[-1].split(".json")[0]
                            json_file = folder_to_save + np.str(json_num) + '.json'

                            # print("og: ", file)
                            # print("json_file: ", json_file)

                            camera_settings = camera.split("/")[0]
                            mat = json_to_mat(json_file, camera_settings)

                            move_mat_name = folder_to_move + np.str(image_num) + '-meta.mat'
                            sio.savemat(move_mat_name, mat)

                        elif image_ext == ".png":
                            move_file_name = folder_to_move + np.str(image_num) + '_rgb.png'

                        elif image_ext == ".depth.cm.8.png" or image_ext == ".depth.png":
                            move_file_name = folder_to_move + np.str(image_num) + '_depth.png'

                        elif image_ext == ".cs.png":
                            move_file_name = folder_to_move + np.str(image_num) + '_label.png'

                        else:
                            pass

                        shutil.copyfile(old_file_name, move_file_name)

                offset_train += train_files_len
                offset_val += val_files_len
                offset_test += test_files_len
