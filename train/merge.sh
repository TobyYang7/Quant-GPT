#!/bin/bash
# DO NOT use quantized model or quantization_bit when merging lora weights

export USE_MODELSCOPE_HUB=1
MODEL_NAME="v1"

CUDA_VISIBLE_DEVICES=3 python exp_model.py \
    --model_name_or_path  qwen/Qwen-7B-Chat \
    --adapter_name_or_path saves/$MODEL_NAME \
    --template qwen \
    --export_dir  ../exp_model/$MODEL_NAME \
    --export_size 2 \
    --export_legacy_format false \
