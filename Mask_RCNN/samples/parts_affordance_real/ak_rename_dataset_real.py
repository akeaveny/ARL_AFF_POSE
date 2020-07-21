import numpy as np
import shutil
import glob
import os

# # =================== new directory ========================
# folder_to_save = '/data/Akeaveny/Datasets/part-affordance_combined/real/combined_tools/'
# offset = 0
#
# # =================== single objects ========================
# image_dir = '/data/Akeaveny/Datasets/part-affordance_combined/real/part-affordance-tools/tools/'
# images_path1 = image_dir + 'bowl_0*/bowl_0*_*'
# images_path2 = image_dir + 'cup_0*/cup_0*_*'
# images_path3 = image_dir + 'hammer_0*/hammer_0*_*'
# images_path4 = image_dir + 'knife_*/knife_0*_*'
# images_path5 = image_dir + 'ladle_0*/ladle_0*_*'
# images_path6 = image_dir + 'mallet_0*/mallet_0*_*'
# images_path7 = image_dir + 'mug_0*/mug_0*_*'
# images_path8 = image_dir + 'pot_0*/pot_0*_*'
# images_path9 = image_dir + 'saw_0*/saw_0*_*'
# images_path10 = image_dir + 'scissors_0*/scissors_0*_*'
# images_path11 = image_dir + 'scoop_0*/scoop_0*_*'
# images_path12 = image_dir + 'shears_0*/shears_0*_*'
# images_path13 = image_dir + 'shovel_0*/shovel_0*_*'
# images_path14 = image_dir + 'spoon_0*/spoon_0*_*'
# images_path15 = image_dir + 'tenderizer_0*/tenderizer_0*_*'
# images_path16 = image_dir + 'trowel_0*/trowel_0*_*'
# images_path17 = image_dir + 'turner_0*/turner_0*_*'
# image_paths = [images_path1, images_path2, images_path3, images_path4, images_path5,
#                images_path6, images_path7, images_path8, images_path9, images_path10,
#                images_path11, images_path12, images_path13, images_path14, images_path15,
#                images_path16, images_path17]

# =================== new directory ========================
folder_to_save = '/data/Akeaveny/Datasets/part-affordance_combined/real/combined_clutter/'
offset = 0

# =================== single objects ========================
image_dir = '/data/Akeaveny/Datasets/part-affordance_combined/real/part-affordance-clutter/clutter/'
images_path1 = image_dir + 'scene_0*/scene_0*_*'
image_paths = [images_path1]

# =================== images ext ========================
image_ext1 = '_rgb.jpg'
image_ext2 = '_depth.png'
image_ext3 = '_label.mat'
image_exts = [image_ext1, image_ext2, image_ext3]

# =================== new directory ========================
for image_path in image_paths:
    files = None
    for image_ext in image_exts:
        file_path = image_path + image_ext
        print("File path: ", file_path)
        files = sorted(glob.glob(file_path))
        print("Loaded files: ", len(files))

        for idx, file in enumerate(files):
            old_file_name = file

            # image_num = offset + idx
            count = 1000000 + offset + idx
            image_num = str(count)[1:]

            if image_ext == '_rgb.jpg':
                new_file_name = folder_to_save + np.str(image_num) + '_rgb.jpg'
            else:
                new_file_name = folder_to_save + np.str(image_num) + image_ext

            # print("Old File: ", old_file_name)
            # print("New File: ", new_file_name)

            shutil.copyfile(old_file_name, new_file_name)

    offset += len(files)
    print("offset:", offset)

