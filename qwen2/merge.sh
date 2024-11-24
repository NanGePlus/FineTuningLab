#! /usr/bin/env bash

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=model_merge
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}
mkdir -p $OUTPUT_DIR

MODEL_DIR="/root/autodl-tmp/Qwen2.5-7B-Instruct"
CHECKPOINT_DIR="output/model_train-20241103-132703/checkpoint-2150"

CUDA_VISIBLE_DEVICES=0 python merge.py \
  --model $MODEL_DIR \
  --ckpt $CHECKPOINT_DIR \
  --output_dir $OUTPUT_DIR




