MODEL_DIR="../qwen2/output/model_merge-20241118-113242"

CUDA_VISIBLE_DEVICES=0 python webui_qwen2.py \
    --model $MODEL_DIR
