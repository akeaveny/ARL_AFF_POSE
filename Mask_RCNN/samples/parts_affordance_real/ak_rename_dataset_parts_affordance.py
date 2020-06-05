import numpy as np
import shutil
import glob
import os

# =================== new directory ========================
folder_to_save = '/data/Akeaveny/Datasets/part-affordance-dataset/ndds_and_real/Kitchen_Knife_val_real5/'
# folder_to_save = '/data/Akeaveny/Datasets/part-affordance-dataset/real/part-affordance-clutter/clutter_combined/'
offset = 0

# =================== single objects ========================
images_path1 = '/data/Akeaveny/Datasets/part-affordance-dataset/ndds_and_real/Kitchen_Knife_val_real5/*'
# images_path2 = '/data/Akeaveny/Datasets/part-affordance-dataset/real/part-affordance-dataset/tools/cup_0*/cup_0*_*'
# images_path3 = '/data/Akeaveny/Datasets/part-affordance-dataset/real/part-affordance-dataset/tools/hammer_0*/hammer_0*_*'
# images_path4 = '/data/Akeaveny/Datasets/part-affordance-dataset/real/part-affordance-dataset/tools/knife_*/knife_*_00000[0-1]??'
# images_path4 = '/data/Akeaveny/Datasets/part-affordance-dataset/real/part-affordance-dataset/tools/knife_*/knife_*_000002[0-3]?'
# images_path5 = '/data/Akeaveny/Datasets/part-affordance-dataset/real/part-affordance-dataset/tools/ladle_0*/ladle_0*_*'
# images_path6 = '/data/Akeaveny/Datasets/part-affordance-dataset/real/part-affordance-dataset/tools/mallet_0*/mallet_0*_*'
# images_path7 = '/data/Akeaveny/Datasets/part-affordance-dataset/real/part-affordance-dataset/tools/mallet_0*/mug_0*_*'
# images_path8 = '/data/Akeaveny/Datasets/part-affordance-dataset/real/part-affordance-dataset/tools/saw_0*/saw_0*_*'
# images_path9 = '/data/Akeaveny/Datasets/part-affordance-dataset/real/part-affordance-dataset/tools/scoop_0*/scoop_0*_*'
# images_path10 = '/data/Akeaveny/Datasets/part-affordance-dataset/real/part-affordance-dataset/tools/shears_0*/shears_0*_*'
# images_path11 = '/data/Akeaveny/Datasets/part-affordance-dataset/real/part-affordance-dataset/tools/spoon_0*/spoon_0*_*'
# images_path12 = '/data/Akeaveny/Datasets/part-affordance-dataset/real/part-affordance-dataset/tools/tenderizer_0*/tenderizer_0*_*'
# images_path13 = '/data/Akeaveny/Datasets/part-affordance-dataset/real/part-affordance-dataset/tools/trowel_0*/trowel_0*_*'
# images_path14 = '/data/Akeaveny/Datasets/part-affordance-dataset/real/part-affordance-dataset/tools/turner_0*/turner_0*_*'
image_paths = [images_path1]

# =================== images ext ========================
image_ext1 = '_rgb.jpg'
image_ext2 = '_depth.png'
image_ext3 = '_label.mat'
# image_ext4 = '.cs.png'
# image_ext5 = '.depth.16.png'
# image_ext6 = '.json'
# image_ext7 = '.png'
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

            # object = file.split("clutter/")[1]
            # object = "." + object.split("_")[0] + "."
            # print(object)

            # image_num = offset + idx
            count = 100000 + offset + idx
            image_num = str(count)[1:]

            if image_ext == '_rgb.jpg':
                new_file_name = folder_to_save + np.str(image_num) + '_rgb.jpg'
            # elif image_ext == ".png":
            #     new_file_name = folder_to_save + np.str(image_num) + '_rgb.png'
            # elif image_ext == ".cs.png":
            #     new_file_name = folder_to_save + np.str(image_num) + '_label.png'
            # elif image_ext == ".depth.16.png":
            #     new_file_name = folder_to_save + np.str(image_num) + '_depth.png'
            else:
                # new_file_name = folder_to_save + np.str(image_num) + image_ext.split(".")[0] + object + image_ext.split(".")[1]
                new_file_name = folder_to_save + np.str(image_num) + image_ext

            print("Old File: ", old_file_name)
            print("New File: ", new_file_name)

            shutil.copyfile(old_file_name, new_file_name)

    offset += len(files)

