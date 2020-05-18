#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=1

python3 ./tools/train_pringles.py --dataset pringles\
  --dataset_root /data/Akeaveny/Datasets/pringles/zed \

