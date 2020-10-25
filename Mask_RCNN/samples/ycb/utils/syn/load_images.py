import cv2
import glob
import numpy as np

import argparse

############################################################
#  argparse
############################################################
parser = argparse.ArgumentParser(description='Load YCB Images')

parser.add_argument('--dataset', required=False, default='/data/Akeaveny/Datasets/YCB_Video_Dataset/data_combined/',
                    type=str,
                    metavar="/path/to/Affordance/dataset/")
parser.add_argument('--split', required=False, default='val/',
                    type=str,
                    metavar='train/ or val/')

args = parser.parse_args()

#########################
# load images
#########################

image_path_ = args.dataset + args.split + "*_rgb.png"
images = sorted(glob.glob(image_path_))

print('{} Split has {} Images'.format(args.split, len(images)))
