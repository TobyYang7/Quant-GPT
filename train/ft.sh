#!/bin/bash

export USE_MODELSCOPE_HUB=1
DATASET_NAME="ft_data_train"
MODEL_NAME="v1"
# MODEL_PATH="/home/zhangmin/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct"
MODEL_PATH="ZhipuAI/chatglm3-6b"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file single_config.yaml \
    run_exp.py \
    --stage sft \
    --do_train \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET_NAME \
    --dataset_dir ./ \
    --template chatglm3 \
    --finetuning_type lora \
    --output_dir ./saves/$MODEL_NAME \
    --overwrite_output_dir \
    --cutoff_len 8192 \
    --vllm_maxlen 8192 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --learning_rate 5e-5 \
    --num_train_epochs 4 \
    --plot_loss \
    --fp16 \
    --load_best_model_at_end \
    --save_strategy epoch \
    --do_eval false \
    --ddp_timeout 180000000 \
    --max_samples 2 \
    # --overwrite_cache \
    # --use_cache false
    # --orpo_beta 0.05 \
    # --quantization_bit 4
    # --lora_target q_proj,v_proj \
    # --streaming \
    # --max_steps 10000
    # --optim paged_adamw_32bit \
    # --report_to tensorboard \
    # --quantization_bit 8 \
    # --flash_attn auto \

