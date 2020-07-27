#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/eval_linemod.py --dataset_root /data/Akeaveny/Datasets/linemod/Linemod_preprocessed\
  --model trained_models/linemod/pose_model_6_0.012692721084131118.pth\
  --refine_model trained_models/linemod/pose_refine_model_87_0.006527408261342158.pth