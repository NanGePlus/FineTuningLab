#! /usr/bin/env bash
MODEL_DIR="/root/autodl-tmp/Qwen2.5-7B-Instruct"
CHECKPOINT_DIR="output/model_train-20241103-132703/checkpoint-2150"

CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --model $MODEL_DIR \
  --ckpt $CHECKPOINT_DIR \
  --data dataset/test.full.jsonl
