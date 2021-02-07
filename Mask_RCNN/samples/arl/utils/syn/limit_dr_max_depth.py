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

MAX_DEPTH = int(10e3)

def dr_limit_max_depth(dr_depth_img):
    rows, cols = dr_depth_img.shape[0], dr_depth_img.shape[1]
    for row in range(rows):
        for col in range(cols):
            if dr_depth_img[row][col] > MAX_DEPTH:
                dr_depth_img[row][col] = MAX_DEPTH
    return dr_depth_img

###########################################################
#
###########################################################
if __name__ == '__main__':

    ######################
    # UMD
    ######################

    ### SYN
    # data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/combined_tools5_'
    #
    # scenes = [
    #     # 'bench/',
    #     # 'dr/',
    #     # 'floor/',
    #     # 'turn_table/' ### MAX DEPTH = 255
    #   ]

    ######################
    # ARL
    ######################

    ### REAL
    # data_path = '/data/Akeaveny/Datasets/arl_dataset/combined_real_tools_4_'
    # data_path = '/data/Akeaveny/Datasets/arl_dataset/combined_real_clutter_4_'

    ### SYN
    data_path = '/data/Akeaveny/Datasets/arl_dataset/combined_syn_tools_2_'
    # data_path = '/data/Akeaveny/Datasets/arl_dataset/combined_syn_clutter_2_'

    # 2.
    scenes = [
        # '', ## REAL
        # '1_bench/', ### MAX DEPTH = 5926
        # '2_work_bench/',
        # '3_coffee_table/',
        # '4_old_table/',
        # '5_bedside_table/',
        '6_dr/', ### MAX DEPTH = 43460
    ]

    ######################
    # load from
    ######################

    # 3.
    splits = [
            'test/',
            'train/',
            'val/',
            ]

    # 4.
    image_exts = [
        '.png',
    ]

    ######################
    # loop
    ######################
    num_files = 0
    for split in splits:
        for scene in scenes:
            print('\n*** Split:{}, Scene:{} ***'.format(split, scene))
            file_path = data_path + split + scene + '*_depth.png'
            depth_files = sorted(glob.glob(file_path))
            print("Loaded files: ", len(depth_files))
            num_files += len(depth_files)

            for idx, file in enumerate(depth_files):
                depth = np.array(Image.open(file))
                depth = dr_limit_max_depth(depth)
                max = np.max(depth)
                print(f'Idx: {idx} has max depth: {max}')
                # cv2.imwrite(file, depth.astype(np.uint16))
                ###
                cv2.imshow("depth", np.array(depth, dtype=np.uint8))
                cv2.waitKey(0)



