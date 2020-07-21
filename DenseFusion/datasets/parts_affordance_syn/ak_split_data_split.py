import glob
import numpy as np

data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/combined_tools_train/dr/'
folder_to_save = 'combined_tools_train/dr/'
label_format = '_label.png'

labels = data_path + '??????' + '_label.png'
files = sorted(glob.glob(labels))
print("Loaded files: ", len(files))

# min_img = 0
# max_img = len(images)
# max_img = 400

f_train = open("train_data_list.txt", 'w')
# ===================== train ====================
print('-------- TRAIN --------')
for file in files:

    str_num = file.split(data_path)[1]
    img_number = str_num.split('_label.png')[0]
    print("Img Num: {}/{}".format(img_number, len(files)))
    label_addr = data_path + img_number + label_format

    img_index_str = label_addr.split(folder_to_save)[1]
    img_index_str = img_index_str.split(label_format)[0]
    f_train.write(folder_to_save + img_index_str)
    f_train.write('\n')
f_train.close

data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/combined_tools_val/dr/'
folder_to_save = 'combined_tools_val/dr/'
label_format = '_label.png'

labels = data_path + '??????' + '_label.png'
files = sorted(glob.glob(labels))
print("Loaded files: ", len(files))

# min_img = 0
# max_img = len(images)
# max_img = 400

f_val = open("test_data_list.txt", 'w')
# ===================== val ====================
print('-------- VAL --------')
for file in files:

    str_num = file.split(data_path)[1]
    img_number = str_num.split('_label.png')[0]
    print("Img Num: {}/{}".format(img_number, len(files)))
    label_addr = data_path + img_number + label_format

    img_index_str = label_addr.split(folder_to_save)[1]
    img_index_str = img_index_str.split(label_format)[0]
    f_val.write(folder_to_save + img_index_str)
    f_val.write('\n')
f_val.close

# data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/combined_tools_val/dr/'
# folder_to_save = 'combined_tools_val/dr/'
# label_format = '_label.png'
#
# label_images = []
# label_addrs = data_path + '*_label.png'
# images = [label_images.append(file) for file in sorted(glob.glob(label_addrs))]
# print("\nLoaded Images: ", len(images))
#
# min_img = 0
# max_img = len(images)
# # max_img = 100
#
# iteration = 0
# f_val = open("test_data_list.txt", 'w')
# # ===================== val ====================
# print('-------- VAL --------')
# for i in range(min_img, max_img):
#     print('Image: {}/{}'.format(i, max_img))
#     count = 1000000 + i
#     img_number = str(count)[1:]
#     label_addr = data_path + img_number + label_format
#
#     img_index_str = label_addr.split(folder_to_save)[1]
#     img_index_str = img_index_str.split(label_format)[0]
#     f_val.write(folder_to_save + img_index_str)
#     f_val.write('\n')
# f_val.close

