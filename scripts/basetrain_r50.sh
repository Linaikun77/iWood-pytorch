#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=0
EXP_DIR=exps/basetrain/r50

mkdir -p exps
mkdir -p ${EXP_DIR}

python tools/train.py \
    --config configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    --seed 66 \
    2>&1 | tee ${EXP_DIR}/no11-C90-1.txt
#     2>&1 | tee ${EXP_DIR}/Binary_Classification.txt
#     --tuning weights/no-11/c13-90/no11-C90-checkpoint.pth \
