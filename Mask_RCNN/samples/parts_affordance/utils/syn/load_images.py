import numpy as np
import shutil
import glob
import os

import skimage.io

import matplotlib.pyplot as plt

# Flags
debug = False

###########################################################
#
###########################################################
if __name__ == '__main__':

    ######################
    # dir
    ######################
    data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/combined_tools_'

    ######################
    # load from
    ######################

    # 2.
    scenes = [
              # 'turn_table/', 'bench/', 'floor/',
              'dr/'
              ]

    # 3.
    splits = [
            'val/',
            'train/',
            ]

    # 4.
    image_ext10 = '_rgb.png'
    image_exts1 = [
        image_ext10,
    ]

    ######################
    # loop
    ######################
    num_files = 0
    for split in splits:
        for scene in scenes:
            print('\n***************** {} *****************'.format(scene))
            rgb_file_path = data_path + split + scene + '??????' + '_rgb.png'
            rgb_files = sorted(glob.glob(rgb_file_path))
            print("Loaded files: ", len(rgb_files))
            num_files += len(rgb_files)
        print("\n-----{} has {} files-----".format(split, num_files))
    print("Dataset has {} files".format(num_files))