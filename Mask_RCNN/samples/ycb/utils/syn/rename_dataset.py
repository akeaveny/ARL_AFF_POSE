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
data_path = '/data/Akeaveny/Datasets/ycb_test/ycb_test_dr/'
new_data_path = '/data/Akeaveny/Datasets/ycb_test/combined_tools_'

####################
# scenes
####################
# 1.
folder_to_object = 'ycb_test_dr/'

objects = [
           ''
            ]

# 2.
scenes = [
        #'bench/', 'floor/', 'turn_table/',
        #'dr/'
        ''
          ]

# 3.
splits = [
          'train/',
          'val/'
          ]

# 4.
cameras = [
    'Kinetic/',
    # 'Xtion/',
    # 'ZED/'
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
    image_ext30,
    image_ext40,
    image_ext50
]

##########################################
# new directory
##########################################
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