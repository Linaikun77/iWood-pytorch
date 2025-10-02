#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
EXP_DIR=exps/eval/r50

mkdir -p exps
mkdir -p ${EXP_DIR}

python tools/train.py \
    --config configs/iwood/iwood_r50vd_6x_bc.yml \
    --seed 66 \
    --resume weights/binary-classifier/checkpoint-bc.pth \
    --test-only \
    2>&1 | tee ${EXP_DIR}/bc-cut-batch3.txt

