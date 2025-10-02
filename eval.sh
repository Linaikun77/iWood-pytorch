#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

EXP_DIR=exps/eval/r50

mkdir -p exps
mkdir -p ${EXP_DIR}

python eval.py \
  --config configs/iwood/iwood_r50vd_6x_bc.yml \
  --image_dir data/iWood-data-cut \
  --output_dir output/iWood-data-cut \
  --resume weights/binary-classifier/checkpoint-bc.pth \
  --print_results \
  2>&1 | tee ${EXP_DIR}/eval_cut.txt
#     --print_output \
#   --plot

