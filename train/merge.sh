#!/bin/bash
# DO NOT use quantized model or quantization_bit when merging lora weights

export USE_MODELSCOPE_HUB=1
MODEL_NAME="v2_14b"

CUDA_VISIBLE_DEVICES=3 python exp_model.py \
    --model_name_or_path  TongyiFinance/Tongyi-Finance-14B-Chat \
    --adapter_name_or_path saves/$MODEL_NAME \
    --template qwen \
    --export_dir  ../exp_model/$MODEL_NAME \
    --export_size 5 \
    --export_legacy_format true \
    # --quantization_bit 4 \
