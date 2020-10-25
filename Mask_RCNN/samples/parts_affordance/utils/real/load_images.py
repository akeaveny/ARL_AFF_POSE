import numpy as np
import shutil
import glob
import os

import skimage.io
from PIL import Image
import matplotlib.pyplot as plt

# Flags
debug = False

###########################################################
#
###################################################a########
if __name__ == '__main__':

    ######################
    # dir
    ######################
    data_path = '/data/Akeaveny/Datasets/part-affordance_combined/real/combined_tools_test_scissors/'
    file_path = data_path + '*' + '_label.png'
    files = sorted(glob.glob(file_path))
    print("Dataset has {} files".format(len(files)))

    for file in files:
        plt.plot()
        plt.imshow(np.arange(Image.open(file)))
        plt.show()
        plt.ioff()