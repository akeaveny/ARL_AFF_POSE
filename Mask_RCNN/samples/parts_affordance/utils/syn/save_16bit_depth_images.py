import numpy as np
import shutil
import glob
import os

from PIL import Image

import skimage.io

import matplotlib.pyplot as plt
import cv2

# Flags
debug = False

uint8_MAX_DEPTH = int(255)
uint16_MAX_DEPTH = int(10e3)

# def dr_limit_max_depth(dr_depth_img):
#     rows, cols = dr_depth_img.shape[0], dr_depth_img.shape[1]
#     for row in range(rows):
#         for col in range(cols):
#             if dr_depth_img[row][col] > MAX_DEPTH:
#                 dr_depth_img[row][col] = MAX_DEPTH
#     return dr_depth_img

###########################################################
#
###########################################################
if __name__ == '__main__':

    ######################
    # UMD
    ######################

    ### SYN
    data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/umd_affordance/*/*/*/*/*/'

    scenes = [''
        # 'bench/',
        # 'dr/',
        # 'floor/',
        # 'turn_table/' ### MAX DEPTH = 255
      ]

    ######################
    # load from
    ######################

    # 3.
    splits = [''
            # 'test/',
            # 'train/',
            # 'val/',
            ]

    ######################
    # loop
    ######################
    num_files = 0
    for split in splits:
        for scene in scenes:
            print('\n*** Split:{}, Scene:{} ***'.format(split, scene))
            file_path = data_path + split + scene + '*.depth.png'
            depth_files = sorted(glob.glob(file_path))
            print("Loaded files: ", file_path)
            print("Loaded files: ", len(depth_files))
            num_files += len(depth_files)

            for idx, file in enumerate(depth_files):
                uint8_depth = np.array(Image.open(file), dtype=np.uint16)
                # print(f'Idx: {idx} has max uint8_depth: {np.max(uint8_depth)}')
                uint16_depth = np.array(uint8_depth.copy() * int(uint16_MAX_DEPTH / uint8_MAX_DEPTH), dtype=np.uint16)
                # print(f'Idx: {idx} has max uint16_depth: {np.max(uint16_depth)}')

                ###
                new_file = file.split('.depth.png')[0] + '.depth.16.png'
                cv2.imwrite(new_file, uint16_depth.astype(np.uint16))

                ###
                # cv2.imshow("uint8_depth", np.array(uint8_depth, dtype=np.uint8))
                # cv2.imshow("uint16_depth", np.array(uint16_depth, dtype=np.uint8))
                # cv2.waitKey(0)



