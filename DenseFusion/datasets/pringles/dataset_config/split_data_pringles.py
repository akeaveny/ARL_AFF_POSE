import glob
import numpy as np


f_train = open("train_data_list.txt", 'w')
f_val = open("test_data_list.txt", 'w')
# ============== config =====================
train_images = ['000000', '000003', '000006', '000008']
val_images = ['000002', '000005', '000009']

# ============= load labels ==================
label_images = []
label_addrs = '/data/Akeaveny/Datasets/pringles/Alex/train/images/*.cs.png'
images = [label_images.append(file) for file in glob.glob(label_addrs)]
print("Loaded Images: ", len(images))

# ================ test ===========
train_addrs = []
data_addrs = '/data/Akeaveny/Datasets/pringles/Alex/train/images/'
for test_image in train_images:
    temp = data_addrs + test_image + '.cs.png'
    train_addrs.append(temp)
# print(test_addrs)

for label_image in label_images:
    if label_image in train_addrs:
        # print(label_image)
        img_index_str = label_image.split("Alex/")[1]
        img_index_str = img_index_str.split(".cs")[0]
        f_train.write(img_index_str)
        f_train.write('\n')

# ================ test ===========
val_addrs = []
data_addrs = '/data/Akeaveny/Datasets/pringles/Alex/train/images/'
for val_image in val_images:
    temp = data_addrs + val_image + '.cs.png'
    val_addrs.append(temp)
# print(test_addrs)

for label_image in label_images:
    if label_image in val_addrs:
        # print(label_image)
        img_index_str = label_image.split("Alex/")[1]
        img_index_str = img_index_str.split(".cs")[0]
        f_val.write(img_index_str)
        f_val.write('\n')
