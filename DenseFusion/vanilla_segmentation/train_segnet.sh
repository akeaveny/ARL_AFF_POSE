#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 train.py \
  --dataset_root /data/Akeaveny/Datasets/YCB_Video_Dataset \
  --resume_model model_current.pth

