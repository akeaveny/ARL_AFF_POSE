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
new_data_path = '/data/Akeaveny/Datasets/ycb_syn/combined_clutter2_'

####################
# scenes
####################
# 1.
folder_to_object = 'ycb_affordance/'

objects = [
        '099_mixed_1/',
        '099_mixed_2/',
        '099_mixed_3/',
        '099_mixed_4/',
        '099_mixed_5/',
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
for split in splits:
    offset_train, offset_val = 0, 0
    for object in objects:
        for scene in scenes:
            for camera in cameras:
                files_offset = 0
                for image_ext in image_exts1:
                    file_path = data_path + object + scene + split + camera + '??????' + image_ext
                    print("File path: ", file_path)
                    files = np.array(sorted(glob.glob(file_path)))
                    print("Loaded files: ", len(files))
                    print("offset: ", offset_train, offset_val)

                    ###############
                    # split files
                    ###############
                    np.random.seed(0)
                    total_idx = np.arange(0, len(files), 1)
                    train_idx = np.random.choice(total_idx, size=int(train_val_split * len(files)), replace=False)
                    val_idx = np.delete(total_idx, train_idx)
                    train_files = files[train_idx]
                    val_files = files[val_idx]
                    print("Chosen Train Files {}/{}".format(len(train_files), len(files)))
                    # print("Train Idx ", train_idx)
                    print("Chosen Val Files {}/{}".format(len(val_files), len(files)))
                    # print("Val Idx ", val_idx)

                    ###############
                    # train
                    ###############
                    split_folder =  'train/'

                    for idx, file in enumerate(train_files):
                        old_file_name = file
                        folder_to_save = data_path + object + scene + split + camera
                        folder_to_move = new_data_path + split_folder + scene

                        # image_num = offset + idx
                        count = 1000000 + offset_train + idx
                        image_num = str(count)[1:]

                        if image_ext == ".json":
                            new_file_name = folder_to_save + np.str(image_num) + '.json'
                            move_file_name = folder_to_move + np.str(image_num) + '.json'

                            json_num = file.split("/")[-1].split(".json")[0]
                            json_file = folder_to_save + np.str(json_num) + '.json'

                            print("og: ", file)
                            print("json_file: ", json_file)

                            camera_settings = camera.split("/")[0]
                            mat = json_to_mat(json_file, camera_settings)

                            new_mat_name = folder_to_save + np.str(image_num) + '-meta.mat'
                            sio.savemat(new_mat_name, mat) # TODO:
                            move_mat_name = folder_to_move + np.str(image_num) + '-meta.mat'
                            sio.savemat(move_mat_name, mat)

                        elif image_ext == ".png":
                            new_file_name = folder_to_save + np.str(image_num) + '_rgb.png'
                            move_file_name = folder_to_move + np.str(image_num) + '_rgb.png'

                        elif image_ext == ".depth.cm.8.png" or image_ext == ".depth.png":
                            new_file_name = folder_to_save + np.str(image_num) + '_depth.png'
                            move_file_name = folder_to_move + np.str(image_num) + '_depth.png'

                        elif image_ext == ".cs.png":
                            new_file_name = folder_to_save + np.str(image_num) + '_label.png'
                            move_file_name = folder_to_move + np.str(image_num) + '_label.png'

                        else:
                            pass

                        # print("Old File: ", old_file_name)
                        # print("New File: ", new_file_name)
                        # print("Move File: ", move_file_name)

                        shutil.copyfile(old_file_name, move_file_name)

                    ###############
                    # train
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
                            new_file_name = folder_to_save + np.str(image_num) + '.json'
                            move_file_name = folder_to_move + np.str(image_num) + '.json'

                            json_num = file.split("/")[-1].split(".json")[0]
                            json_file = folder_to_save + np.str(json_num) + '.json'

                            # print("og: ", file)
                            # print("json_file: ", json_file)

                            camera_settings = camera.split("/")[0]
                            mat = json_to_mat(json_file, camera_settings)

                            new_mat_name = folder_to_save + np.str(image_num) + '-meta.mat'
                            sio.savemat(new_mat_name, mat)  # TODO:
                            move_mat_name = folder_to_move + np.str(image_num) + '-meta.mat'
                            sio.savemat(move_mat_name, mat)

                        elif image_ext == ".png":
                            new_file_name = folder_to_save + np.str(image_num) + '_rgb.png'
                            move_file_name = folder_to_move + np.str(image_num) + '_rgb.png'

                        elif image_ext == ".depth.cm.8.png" or image_ext == ".depth.png":
                            new_file_name = folder_to_save + np.str(image_num) + '_depth.png'
                            move_file_name = folder_to_move + np.str(image_num) + '_depth.png'

                        elif image_ext == ".cs.png":
                            new_file_name = folder_to_save + np.str(image_num) + '_label.png'
                            move_file_name = folder_to_move + np.str(image_num) + '_label.png'

                        else:
                            pass

                        # print("Old File: ", old_file_name)
                        # print("New File: ", new_file_name)
                        # print("Move File: ", move_file_name)

                        shutil.copyfile(old_file_name, move_file_name)
                offset_train += len(train_files)
                offset_val += len(val_files)