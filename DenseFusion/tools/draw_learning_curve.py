#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw learning the curve of Densefusion
@author: g13
"""

import os
import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import glob

# =================== argparse ========================
# import argparse
#
# parser = argparse.ArgumentParser(description='Compute Image Mean and Stddev')
# parser.add_argument('--dataset', required=True,
#                     metavar="/path/to//dataset/",
#                     help='Directory of the dataset')
# args = parser.parse_args()
# root_dir = args.dataset

# =================== glob ========================
# log_dir = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/experiments/logs/pringles/epoch_*_log.txt'
# train_files = sorted(glob.glob(log_dir))
#
# log_dir = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/experiments/logs/pringles/epoch_*_test_log.txt'
# test_files = sorted(glob.glob(log_dir))


# =============== DenseFusion config =========================
root_dir = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/experiments/logs/pringles/'
first_id = 1
last_id = 50

save_img_nm = '../pringles_learning_curve_small_batch.png'
train_data_trend = []
test_data_trend = []

# =============== TRAIN DATA =========================
for i in range(first_id, last_id):
    file_name = '{0}/epoch_{1}_log.txt'.format(root_dir, i)
    s = []
    with open(file_name) as f:
        s = f.readlines()[-1]
        x = s.split()
        ave_dist = x[-1].split(":", 1)[1]
        # print(ave_dist)
        f.close()
    train_data_trend.append(ave_dist)
train_dis_trend = np.array(train_data_trend)
train_dis_trend_r = train_dis_trend.astype(np.float).round(5)

# print('--------Train Stats----------')
# max_y = float(max(train_dis_trend))
# min_y = float(min(train_dis_trend))
# #space_size = (min_y+max_y)/10
# #dis_trend_r = dis_trend.round(5)
# print('Train: Min:{:.2f} & Max:{:.2f}'.format(min_y, max_y))

# =============== TEST DATA =========================

for i in range(first_id, last_id):
    file_name = '{0}/epoch_{1}_test_log.txt'.format(root_dir, i)
    # print(file_name)
    s = []
    with open(file_name) as f:
        s = f.readlines()[-1]
        x = s.split()
        # print(x[-1])
        f.close()
    test_data_trend.append(x[-1])
test_dis_trend = np.array(test_data_trend)
test_dis_trend_r = test_dis_trend.astype(np.float).round(5)

# print('--------Test Stats----------')
# max_y = float(max(test_dis_trend))
# min_y = float(min(test_dis_trend))
# #space_size = (min_y+max_y)/10
# #dis_trend_r = dis_trend.round(5)
# print('Train: Min:{:.2f} & Max:{:.2f}'.format(min_y, max_y))

# ============= PLOTTING ============================
fig, ax = plt.subplots()
l1, = ax.plot(range(last_id-1), train_dis_trend_r, '-x', color='blue', linewidth=2)
l2, = ax.plot(range(last_id-1), test_dis_trend_r, '-o', color='red', linewidth=2)
ax.set(xlabel='epoch', ylabel='Avg dis', title='learning curve')

ax.grid()
plt.legend([l1, l2],["Train", "Test"])

plt.xticks(np.arange(first_id-1,last_id+1,40))
plt.yticks(np.arange(0, max(train_dis_trend_r), step=0.01))
plt.ticklabel_format(style='sci', axis='x')

fig.savefig(save_img_nm)
plt.show()

'''
plt.plot(x, y, color='red', linewidth=2.0)
plt.xlabel('epoch')
plt.ylabel('Avg dis')
plt.title('learning curve')
#plt.grid()
plt.show()
plt.savefig("test.png")
'''