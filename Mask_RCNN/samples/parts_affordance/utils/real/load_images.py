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
    data_path = '/data/Akeaveny/Datasets/part-affordance_combined/real/combined_tools/'
    rgb_file_path = data_path + '??????' + '_rgb.jpg'
    rgb_files = sorted(glob.glob(rgb_file_path))
    print("Dataset has {} files".format(len(rgb_files)))