#!/bin/bash
# DO NOT use quantized model or quantization_bit when merging lora weights

# export USE_MODELSCOPE_HUB=1
MODEL_NAME="v4_cp2000"

CUDA_VISIBLE_DEVICES=3 python exp_model.py \
    --model_name_or_path  /home/zhangmin/.cache/modelscope/hub/TongyiFinance/Tongyi-Finance-14B-Chat \
    --adapter_name_or_path /home/zhangmin/toby/Quant-GPT/train/saves/$MODEL_NAME \
    --export_dir  /home/zhangmin/toby/Quant-GPT/exp_model/$MODEL_NAME \
    --export_size 5 \
    --template qwen \
    # --flash_attn off \
    # --export_legacy_format true \
    # --rope_scaling linear \
    # --quantization_bit 4 \
