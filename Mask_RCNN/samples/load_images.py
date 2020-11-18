import numpy as np
import shutil
import glob
import os

from PIL import Image

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

    ### REAL
    # data_path = '/data/Akeaveny/Datasets/part-affordance_combined/real/combined_tools1_'
    # data_path = '/data/Akeaveny/Datasets/part-affordance_combined/real/combined_clutter1_'

    ### SYN
    # data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/combined_tools5_'
    data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/combined_clutter5_'

    ######################
    # load from
    ######################

    # 2.
    scenes = [
              'turn_table/',
              'bench/',
              'floor/',
              'dr/',
              #   ''
              ]

    # 3.
    splits = [
            'test/',
            'train/',
            'val/',
            # ''
            ]

    # 4.
    image_ext10 = '_depth.png'
    # image_ext10 = '_gt_affordance.png'
    image_exts1 = [
        image_ext10,
    ]

    ######################
    # loop
    ######################
    num_files = 0
    for split in splits:
        for scene in scenes:
            print('\n*** Split:{}, Scene:{} ***'.format(split, scene))
            file_path = data_path + split + scene + '*' + image_ext10
            files = sorted(glob.glob(file_path))
            print("Loaded files: ", len(files))
            num_files += len(files)
        # print("Split:{} has {} files".format(split, num_files))
    print("\n***** Dataset has {} files *****".format(num_files))

    # umd_depth_max = 0
    # for file in files:
    #     max = np.max(np.array(Image.open(file)))
    #     if max > umd_depth_max:
    #         umd_depth_max = max
    # print("umd_depth_max: ", umd_depth_max)
    #
    # for file in files:
    #     img = np.array(Image.open(file))
    #     depth = img * (2 ** 8 - 1) / umd_depth_max  ### 8 bit
    #     print("Depth 16 bit: ", np.min(img), np.max(img), img.dtype)
    #     print("Depth 8 bit: ", np.min(depth), np.max(depth), img.dtype)
    #     # print("Mask: ", np.unique(img))
    #     plt.imshow(img)
    #     plt.show()
    #     plt.ioff()


