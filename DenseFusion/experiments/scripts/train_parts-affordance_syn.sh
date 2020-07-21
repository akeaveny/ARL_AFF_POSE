#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/ak_train.py --dataset='parts-affordance_syn' \
  --dataset_root=/data/Akeaveny/Datasets/part-affordance_combined/ndds2 \

