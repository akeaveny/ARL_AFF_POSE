import numpy as np
import shutil
import glob
import os

import skimage.io

import matplotlib.pyplot as plt

# Flags
debug = False

###########################################################
# Fuse RGB and Depth images
###########################################################

def fuse_rgbd(rgb, depth):

    ######################
    # SYNTHETIC
    ######################
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    height_rgb, width_rgb, channels_rgb = rgb.shape
    height_depth, width_depth = depth.shape
    assert height_rgb == height_depth and width_rgb == width_depth

    ######################
    # RGB + D
    ######################

    rgbd = np.zeros(shape=(height_rgb, width_depth, channels_rgb+1), dtype=np.uint8)

    rgbd[:, :, :channels_rgb] = rgb
    rgbd[:, :, -1] = depth

    return np.array(rgbd, dtype=np.uint8)

###########################################################
#
###########################################################
if __name__ == '__main__':

    ######################
    # dir
    ######################
    data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/combined_tools_hammer_'

    ######################
    # load from
    ######################
    # 1.
    folder_to_object = 'combined_tools_hammer_/'

    # 2.
    scenes = [
              'turn_table/', 'bench/', 'floor/',
              'dr/'
              ]

    # 3.
    splits = [
            # 'val/',
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
    for split in splits:
        offset = 0
        for scene in scenes:
            files_offset = 0

            print('\n******************* {} *******************'.format(scene))

            rgb_file_path = data_path + split + scene + '??????' + '_rgb.png'
            depth_file_path = data_path + split + scene + '??????' + '_depth.png'

            rgb_files = sorted(glob.glob(rgb_file_path))
            depth_files = sorted(glob.glob(depth_file_path))

            print("Loaded files: ", len(rgb_files))
            print("offset: ", offset)

            for i in range(len(rgb_files)):
                print("{}/{}".format(i, len(rgb_files)))

                ######################
                # file addr
                ######################

                rgb_addr = rgb_files[i]
                depth_addr = depth_files[i]

                file_num = rgb_files[i].split(data_path + split + scene)[1]
                file_num = file_num.split('_rgb.png')[0]
                rgbd_addr = data_path + split + scene + file_num + '_fused.png'

                if os.path.isfile(rgb_addr) == False:
                    print(" --- RBG does not exist --- ")
                    exit(1)
                if os.path.isfile(depth_addr) == False:
                    print(" --- Depth does not exist --- ")
                    exit(1)

                ######################
                # load images
                ######################

                rgb = np.array(skimage.io.imread(rgb_addr))
                if debug:
                    print("\nrgb_addr: ", rgb_addr)
                    print("RGB: ", rgb.dtype)
                    print("Min:{}, Max:{}".format(np.min(rgb), np.max(rgb)))

                depth = np.array(skimage.io.imread(depth_addr))
                if debug:
                    print("\ndepth_addr: ", depth_addr)
                    print("depth: ", depth.dtype)
                    print("Min:{}, Max:{}".format(np.min(depth), np.max(depth)))

                rgbd = fuse_rgbd(rgb, depth)
                skimage.io.imsave(rgbd_addr, rgbd)
                if debug:
                    print("\nrgbd_addr: ", rgbd_addr)
                    print("rgbd: ", rgbd.dtype)
                    print("Min:{}, Max:{}".format(np.min(rgbd), np.max(rgbd)))

                ### check
                if debug:
                    plt.subplot(3, 1, 1)
                    plt.title("rgb")
                    plt.imshow(rgb)
                    plt.subplot(3, 1, 2)
                    plt.title("depth")
                    plt.imshow(depth)
                    plt.subplot(3, 1, 3)
                    plt.title("rgbd")
                    plt.imshow(rgbd)
                    plt.show()

            offset += len(rgb_files)