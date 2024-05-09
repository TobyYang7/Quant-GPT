#!/bin/bash
# DO NOT use quantized model or quantization_bit when merging lora weights

export USE_MODELSCOPE_HUB=1
MODEL_NAME="v3_14b"

CUDA_VISIBLE_DEVICES=3 python exp_model.py \
    --model_name_or_path  /home/zhangmin/.cache/modelscope/hub/TongyiFinance/Tongyi-Finance-14B-Chat-llamafy \
    --adapter_name_or_path /home/zhangmin/toby/Quant-GPT/exp_model/v3_adapter/checkpoint-600 \
    --export_dir  /home/zhangmin/toby/Quant-GPT/exp_model/$MODEL_NAME \
    --export_size 30 \
    --export_legacy_format true \
    --shift_attn true \
    --rope_scaling linear \
    --template qwen
    # --quantization_bit 4 \
