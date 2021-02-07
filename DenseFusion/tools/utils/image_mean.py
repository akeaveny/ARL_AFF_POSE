import cv2
import glob
import numpy as np

from PIL import Image

import argparse

############################################################
#  argparse
############################################################
parser = argparse.ArgumentParser(description='Evaluate trained model for DenseFusion')

parser.add_argument('--dataset', required=False, default='/data/Akeaveny/Datasets/elevator_dataset/',
                    type=str,
                    metavar="")
parser.add_argument('--dataset_config', required=False,
                    default='/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/elevator/dataset_config/',
                    type=str,
                    metavar="")

args = parser.parse_args()

#########################
# load images
#########################
num_random = 100
images_file = 'data_lists/elevator_val_data_list.txt'

images_paths = np.loadtxt('{}/{}'.format(args.dataset_config, images_file), dtype=np.str)
print("Num Images: ", len(images_paths))

random_idx = np.random.choice(np.arange(0, len(images_paths), 1), size=int(num_random), replace=False)
print("Random Img Selected: ", len(images_paths[random_idx]))

dataset_mean, dataset_std, dataset_count = 0, 0, 0
for images_path in images_paths[random_idx]:
    image_path_ = args.dataset + images_path + "_rgb.png"
    # print("image_path_: ", image_path_)
    # image = cv2.imread(image_path_)
    pil_image = Image.open(image_path_)
    image = cv2.cvtColor(np.array(pil_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    mean, stddev = cv2.meanStdDev(image.astype(np.uint8))
    # mean, stddev = cv2.meanStdDev(image)
    # print("mean: ", mean)
    dataset_mean += mean
    dataset_std += stddev
    dataset_count += 1
    ### print("{}/{}".format(dataset_count, len(images_paths)))

dataset_mean /= dataset_count
dataset_std /= dataset_count
print("**************** dataset stats (in reverse) ****************")
print("Means (RGB): ",dataset_mean[2], dataset_mean[1], dataset_mean[0])
print("std: ",dataset_std[2], dataset_std[1], dataset_std[0])
#print("Means normalized: \n", dataset_mean[0]/255)
#print("std normalized: \n", dataset_std[0]/255)
