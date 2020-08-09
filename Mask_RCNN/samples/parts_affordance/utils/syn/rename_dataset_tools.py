import numpy as np
import shutil
import glob
import os

from json_to_mat import json_to_mat
import scipy.io as sio

# =================== new directory ========================
# 0.
data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/parts_affordance1/'
new_data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/combined_tools_'

# =================== load from ========================
# 1.
folder_to_object = 'parts_affordance1/'

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
        'bench/', 'floor/', 'turn_table/',
        'dr/'
          ]

# 3.
splits = [
          # 'train/',
          'val/'
          ]

# 4.
cameras = [
    'Kinetic/',
    'Xtion/',
    'ZED/'
]

# =================== images ext ========================
image_ext10 = '.json'
image_ext20 = '.cs.png'
image_ext30 = '.depth.cm.8.png'
image_ext40 = '.depth.png'
image_ext50 = '.png'
image_exts1 = [
    image_ext10,
    image_ext20,
    image_ext30,
    image_ext40,
    image_ext50
]

# =================== new directory ========================
for split in splits:
    offset = 0
    for object in objects:
        for scene in scenes:
            for camera in cameras:
                files_offset = 0
                for image_ext in image_exts1:
                    file_path = data_path + object + scene + split + camera + '??????' + image_ext
                    print("File path: ", file_path)
                    files = sorted(glob.glob(file_path))
                    print("Loaded files: ", len(files))
                    print("offset: ", offset)

                    for idx, file in enumerate(files):
                        old_file_name = file
                        folder_to_save = data_path + object + scene + split + camera
                        folder_to_move = new_data_path + split + '/' + scene

                        # image_num = offset + idx
                        count = 1000000 + offset + idx
                        image_num = str(count)[1:]

                        if image_ext == ".json":
                            new_file_name = folder_to_save + np.str(image_num) + '.json'
                            move_file_name = folder_to_move + np.str(image_num) + '.json'

                            count = 1000000 + idx
                            json_num = str(count)[1:]
                            json_file = folder_to_save + np.str(json_num) + '.json'

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

                        shutil.copyfile(old_file_name, move_file_name)

                offset += len(files)