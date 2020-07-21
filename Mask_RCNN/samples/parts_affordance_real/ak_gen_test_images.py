import numpy as np
import shutil
import glob
import os
#
# # # =================== new directory ========================
# folder_to_save = '/data/Akeaveny/Datasets/part-affordance_combined/real/combined_tools_test/'
# offset = 0
#
# # # =================== single objects ========================
# images_path1 = '/data/Akeaveny/Datasets/part-affordance_combined/real/combined_tools/'
# image_paths = [image_path1]
#
# =================== images ext ========================
image_ext10 = '_label.png'
image_ext20 = '_rgb.jpg'
image_ext30 = '_depth.png'
image_exts = [image_ext10, image_ext20, image_ext30]

# ===================== LOAD DATA ====================
data_path = '/data/Akeaveny/Datasets/part-affordance_combined/real/combined_tools/'
new_data_path = '/data/Akeaveny/Datasets/part-affordance_combined/real/combined_tools_test_hammer/'

# ============== split into test data ===========
min_img = 3788 + 1
max_img = 4904 - 1
idx = np.arange(min_img, max_img, 1)
test_idx = np.random.choice(idx, size=int(100), replace=False)

iteration = 0
for i in test_idx:
    for image_ext in image_exts:

        print('\nImage: {}/{}'.format(iteration, len(image_exts) * np.squeeze(test_idx.shape)))
        count = 1000000 + i
        img_number = str(count)[1:]

        old_file_name = data_path + img_number + image_ext
        new_file_name = new_data_path + img_number + image_ext

        print("Old File: ", old_file_name)
        print("New File: ", new_file_name)

        shutil.copyfile(old_file_name, new_file_name)
        iteration += 1

