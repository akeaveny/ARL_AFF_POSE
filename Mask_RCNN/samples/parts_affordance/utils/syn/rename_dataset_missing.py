import numpy as np
import shutil
import glob
import os

# =================== new directory ========================
# 0.
data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/parts_affordance1/'
new_data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/combined_missing/'

# =================== load from ========================
# 1.
folder_to_object = 'parts_affordance1/'

# 1.
object1 = '_missing/'
objects = [object1]

# 2.
scenes = [
          'turn_table/', 'bench/', 'floor/',
          'dr/'
          ]

# 3.
# splits = ['train/', 'val/', 'test/']

# 4.
cameras = ['Kinetic/']

# =================== images ext ========================
image_ext10 = '.json'
image_ext20 = '.cs.png'
image_ext30 = '.depth.png'
image_ext40 = '.png'
image_exts1 = [ image_ext10, image_ext20, image_ext30, image_ext40]

# =================== new directory ========================
offset = 0
for object in objects:
    for scene in scenes:
        for camera in cameras:
            files_offset = 0
            for image_ext in image_exts1:
                file_path = data_path + object + scene + 'train/' + camera + '??????' + image_ext
                print("File path: ", file_path)
                files = sorted(glob.glob(file_path))
                print("Loaded files: ", len(files))
                print("offset: ", offset)

                for idx, file in enumerate(files):
                    old_file_name = file
                    folder_to_save = data_path + object + camera
                    folder_to_move = new_data_path

                    # image_num = offset + idx
                    count = 1000000 + offset + idx
                    image_num = str(count)[1:]

                    if image_ext == ".json":
                        new_file_name = folder_to_save + np.str(image_num) + '.json'
                        move_file_name = folder_to_move + np.str(image_num) + '.json'

                    elif image_ext == ".png":
                        new_file_name = folder_to_save + np.str(image_num) + '_rgb.png'
                        move_file_name = folder_to_move + np.str(image_num) + '_rgb.png'

                    elif image_ext == ".depth.png":
                        new_file_name = folder_to_save + np.str(image_num) + '_depth.png'
                        move_file_name = folder_to_move + np.str(image_num) + '_depth.png'

                    elif image_ext == ".cs.png":
                        new_file_name = folder_to_save + np.str(image_num) + '_label.png'
                        move_file_name = folder_to_move + np.str(image_num) + '_label.png'

                    else:
                        pass

                    # print("Old File: ", old_file_name)
                    # print("New File: ", move_file_name)

                    shutil.copyfile(old_file_name, move_file_name)
                    ## os.rename(old_file_name, new_file_name)

            offset += len(files)