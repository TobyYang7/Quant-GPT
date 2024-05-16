#!/bin/bash

# export MODELSCOPE_CACHE='/root/autodl-tmp'
export USE_MODELSCOPE_HUB=1
export CUDA_VISIBLE_DEVICES=2,3
DATASET_NAME="sentiment"
MODEL_NAME="v7"
MODEL_PATH="qwen/Qwen-7B-Chat"
# MODEL_PATH="TongyiFinance/Tongyi-Finance-14B-Chat"

deepspeed --num_gpus 2 run_exp.py \
    --stage sft \
    --do_train \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET_NAME \
    --dataset_dir ./ \
    --template 	qwen \
    --finetuning_type lora \
    --output_dir ../exp_model/$MODEL_NAME \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 8196 \
    --vllm_maxlen 8196 \
    --preprocessing_num_workers 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --warmup_ratio 0.05 \
    --evaluation_strategy steps \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --plot_loss \
    --fp16 \
    --load_best_model_at_end \
    --orpo_beta 0.05 \
    --do_eval false \
    --deepspeed ds_z3_config.json \
    --ddp_timeout 180000000 \
    --eval_steps 500 \
    --save_steps 500 \
    --val_size 0.05 \
    --max_samples 1000
    # --shift_attn
    # --weight_decay 0.01 \
    # --max_samples 100 \
    # --print_param_status \
    # --max_grad_norm 1.0 \
    # --max_samples 10 \
    # --quantization_bit 4 \       
    # --streaming \
    # --report_to tensorboard \
    # --quantization_bit 8 \
