#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=0
EXP_DIR=exps/basetrain/r50

mkdir -p exps
mkdir -p ${EXP_DIR}

python tools/train.py \
    --config configs/iwood/iwood_r50vd_6x_coco.yml \
    --seed 66 \
    2>&1 | tee ${EXP_DIR}/no11-C90-1.txt

