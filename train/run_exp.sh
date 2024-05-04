#!/bin/bash

#  export HF_HUB_ENABLE_HF_TRANSFER=1 
#  export HF_ENDPOINT=https://hf-mirror.com
 export USE_MODELSCOPE_HUB=1

DATASET_NAME="test"
MODEL_NAME="v1"


CUDA_VISIBLE_DEVICES=0  python run_exp.py \
    --model_name_or_path ../exp_model/$MODEL_NAME \
    --dataset $DATASET_NAME \
    --dataset_dir ./ \
    --template qwen \
    --output_dir ../saves/exp/$MODEL_NAME/$DATASET_NAME \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 8192 \
    --per_device_train_batch_size 1 \
    --max_new_tokens 200 \
    --temperature 0.5 \
    --do_predict \
    --predict_with_generate \
    --per_device_eval_batch_size 8 \
    --max_samples 3000 \
    # --split test \
    # --adapter_name_or_path $MODEL_PATH \
