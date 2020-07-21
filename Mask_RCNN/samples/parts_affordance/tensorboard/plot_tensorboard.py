import os

import numpy as np
import matplotlib.pyplot as plt
import csv

csv_file = 'train/bbox_loss/run-parts_affordance_combined_parts_affordance_real_small_batch_700_1-tag-mrcnn_bbox_loss.csv'

def read_datafile(file_name):
    # the skiprows keyword is for heading, but I don't know if trailing lines
    # can be specified
    data = np.loadtxt(file_name, delimiter=',', skiprows=10)
    return data

data = read_datafile(csv_file)

time = data[:, 0]
epoch = data[:, 1]
X = data[:, 2]

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.set_title("Mains power stability")
ax1.set_xlabel('epoch')
ax1.set_ylabel('Loss')
ax1.plot(epoch, X, c='r', label='the data')
leg = ax1.legend()
plt.show()