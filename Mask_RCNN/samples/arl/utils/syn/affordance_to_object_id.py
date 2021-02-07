import numpy as np
import shutil
import glob
import os

import skimage.io

import matplotlib.pyplot as plt

VIS = False

###########################################################
# LOOKUP FROM OBJECT ID TO AFFORDANCE LABEL
###########################################################

def seq_get_masks(og_mask):

    instance_masks = np.zeros((og_mask.shape[0], og_mask.shape[1]), dtype=np.uint8)
    instance_mask_one = np.ones((og_mask.shape[0], og_mask.shape[1]), dtype=np.uint8)

    object_id_labels = np.unique(og_mask)
    # print("GT Object ID:", np.unique(object_id_labels))

    for i, object_id in enumerate(object_id_labels):
        if object_id != 0:
            affordance_id = map_affordance_label(object_id)
            # print("Affordance Label:", affordance_id)

            instance_mask = instance_mask_one * affordance_id
            instance_masks = np.where(og_mask==object_id, instance_mask, instance_masks).astype(np.uint8)

            # idx = np.where(og_mask == object_id)[0]
            # instance_masks[idx] = affordance_id

    return instance_masks.astype(np.uint8)

def map_affordance_label(current_id):

    # 1
    mallet = [
        1, # 'mallet_1_grasp'
        2, # 'mallet_4_pound'
    ]

    spatula = [
        3,  # 'spatula_1_grasp'
        4,  # 'spatula_2_support'
    ]

    wooden_spoon = [
        5,  # 'wooden_spoon_1_grasp'
        6,  # 'wooden_spoon_3_scoop'
    ]

    screwdriver = [
        7,  # 'screwdriver_1_grasp'
        8,  # 'screwdriver_2_screw'
    ]

    garden_shovel = [
        9,  # 'garden_shovel_1_grasp'
        10,  # 'garden_shovel_3_scoop'
    ]

    if current_id in mallet:
        return 1
    elif current_id in spatula:
        return 2
    elif current_id in wooden_spoon:
        return 3
    elif current_id in screwdriver:
        return 4
    elif current_id in garden_shovel:
        return 5
    else:
        print(" --- Object ID does not map to Affordance Label --- ")
        exit(1)

###########################################################
#
###########################################################
if __name__ == '__main__':

    data_paths = [
            '/data/Akeaveny/Datasets/arl_dataset/combined_syn_tools_2_',
            '/data/Akeaveny/Datasets/arl_dataset/combined_syn_clutter_2_',
                  ]

    ######################
    # TOOLS
    ######################

    # data_path = '/data/Akeaveny/Datasets/arl_dataset/combined_syn_tools_2_'

    ######################
    # CLUTTER
    ######################

    # data_path = '/data/Akeaveny/Datasets/arl_dataset/combined_syn_clutter_2_'

    ######################
    ######################

    scenes = [
            '1_bench/',
            '2_work_bench/',
            '3_coffee_table/',
            '4_old_table/',
            '5_bedside_table/',
            '6_dr/'
    ]

    splits = [
            'train/',
            'val/',
            'test/',
            ]

    image_ext10 = '_label.png'
    image_exts1 = [
        image_ext10,
    ]

    ######################
    ######################
    for data_path in data_paths:
        for split in splits:
            offset = 0
            for scene in scenes:
                for image_ext in image_exts1:
                    file_path = data_path + split + scene + '*' + image_ext
                    print("File path: ", file_path)
                    files = sorted(glob.glob(file_path))
                    print("Loaded files: ", len(files))
                    print("offset: ", offset)

                    for file in files:

                        object_id_label = np.array(skimage.io.imread(file))
                        affordance_label = seq_get_masks(object_id_label)
                        # print("Affordance_label:", np.unique(affordance_label))

                        filenum = file.split(data_path + split + scene)[1]
                        filenum = filenum.split(image_ext)[0]

                        object_id_file = data_path + split + scene + filenum + '_object_id.png'
                        skimage.io.imsave(object_id_file, affordance_label)

                        # print(file)
                        # print(object_id_file)

                        if VIS:
                            print("object_id_label: ", np.unique(object_id_label))
                            print("affordance_label: ", np.unique(affordance_label))
                            ### plot
                            plt.subplot(2, 1, 1)
                            plt.title("og")
                            plt.imshow(object_id_label*255/2)
                            plt.subplot(2, 1, 2)
                            plt.title("affordance")
                            plt.imshow(affordance_label * 255/2)
                            plt.show()

                offset += len(files)