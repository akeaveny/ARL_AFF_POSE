import cv2
import glob
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import argparse

import skimage.io

############################################################
#  argparse
############################################################
parser = argparse.ArgumentParser(description='Load YCB Images')

parser.add_argument('--ycb_dataset', required=False, default='/data/Akeaveny/Datasets/YCB_Video_Dataset/data/0000/',
                    type=str,
                    metavar="/path/to/Affordance/dataset/")
parser.add_argument('--linemod_dataset', required=False, default='/data/Akeaveny/Datasets/linemod/Linemod_preprocessed/data/01/depth/',
                    type=str,
                    metavar="/path/to/Affordance/dataset/")

parser.add_argument(
                    # '--arl_syn_dataset', required=False,default='/data/Akeaveny/Datasets/arl_scanned_objects/ARL/arl/hammer/turn_table/train/Kinect/',
                    # '--arl_syn_dataset', required=False, default='/data/Akeaveny/Datasets/arl_scanned_objects/ARL/combined_tools_test_depth_val/dr/',
                    '--arl_syn_dataset', required=False, default='/data/Akeaveny/Datasets/arl_scanned_objects/ARL/combined_tools_val/dr/',
                    type=str,
                    metavar="/path/to/Affordance/dataset/")

parser.add_argument(
                    '--umd_syn_dataset', required=False, default='/data/Akeaveny/Datasets/part-affordance_combined/ndds4/umd_affordance1/hammer_01/dr/train/Kinect/',
                    type=str,
                    metavar="/path/to/Affordance/dataset/")

parser.add_argument(
                    # '--umd_real_dataset', required=False, default='/data/Akeaveny/Datasets/part-affordance_combined/real/part-affordance-tools/tools/hammer_01/',
                    '--umd_real_dataset', required=False, default='/data/Akeaveny/Datasets/part-affordance_combined/real/combined_tools1_val/',
                    type=str,
                    metavar="/path/to/Affordance/dataset/")

parser.add_argument(
                    '--ycb_syn_dataset', required=False, default='/data/Akeaveny/Datasets/ycb_syn/ycb_affordance/002_master_chef_can/dr/train/Kinect/',
                    type=str,
                    metavar="/path/to/Affordance/dataset/")

args = parser.parse_args()

#########################
# load images
#########################
# depth_image_ext = "-depth.png"  # ycb
# depth_image_ext = ".png"  # linemod

# depth_image_ext = "_depth.png"
# depth_image_ext = ".depth.mm.16.png"
depth_image_ext = ".depth.png"

# ycb_path_ = args.ycb_dataset + "*" + depth_image_ext
# ycb_files = sorted(glob.glob(ycb_path_))
#
# linemod_path_ = args.linemod_dataset + "*" + '.png'
# linemod_files = sorted(glob.glob(linemod_path_))

arl_syn_path_ = args.ycb_syn_dataset + "*" + depth_image_ext
arl_syn_files = sorted(glob.glob(arl_syn_path_))
arl_syn_max_depth = -np.inf
arl_syn_min_depth = np.inf
NDDS_DEPTH_CONST = 10e3 / (2 ** 8 - 1)

print('Loaded {} Images'.format(len(arl_syn_files)))

for idx, image_addr in enumerate(arl_syn_files):

    # # # load images
    # ycb_depth = np.array(Image.open(ycb_files[0]))
    # linemod_depth = np.array(Image.open(linemod_files[0]))
    # arl_syn_depth = np.array(Image.open(arl_syn_files[0]))
    #
    # print("YCB Depth:\t\t\t dtype: {}, min: {}, max: {}".format(ycb_depth.dtype, np.min(ycb_depth), np.max(ycb_depth)))
    # print("Linemod Depth:\t\t dtype: {}, min: {}, max: {}, shape {}".format(linemod_depth.dtype, np.min(linemod_depth), np.max(linemod_depth), linemod_depth.shape))
    # print("ARL Syn Depth:\t\t dtype: {}, min: {}, max: {}, shape {}".format(arl_syn_depth.dtype, np.min(arl_syn_depth), np.max(arl_syn_depth), arl_syn_depth.shape))
    #
    # ### plot
    # plt.figure(0)
    # plt.subplot(1, 3, 1)
    # plt.title("YCB")
    # plt.imshow(ycb_depth)
    # plt.subplot(1, 3, 2)
    # plt.title("Linemod")
    # plt.imshow(linemod_depth)
    # plt.subplot(1, 3, 3)
    # plt.title("ARL Syn")
    # plt.imshow(arl_syn_depth)
    # plt.show()
    # plt.ioff()

    #####################
    # MAX DEPTH
    #####################
    arl_syn_depth = np.array(Image.open(image_addr)) * NDDS_DEPTH_CONST

    img_max_depth = np.max(arl_syn_depth)
    img_min_depth = np.min(arl_syn_depth)

    # if img_max_depth == (2 ** 16 - 1):
    #     print(image_addr)
        # print("Img dtype:\t{}, Min Depth:\t{}, Max Depth:\t{}".format(arl_syn_depth.dtype, img_min_depth, img_max_depth))
        ### plot
        # plt.figure(0)
        # plt.title("Max Depth")
        # plt.imshow(arl_syn_depth)
        # plt.show()
        # plt.ioff()


    arl_syn_max_depth = img_max_depth if img_max_depth > arl_syn_max_depth else arl_syn_max_depth
    arl_syn_min_depth = img_min_depth if img_min_depth < arl_syn_min_depth else arl_syn_min_depth
print("Min Depth:\t{}, Max Depth:\t{}".format(arl_syn_min_depth, arl_syn_max_depth))







