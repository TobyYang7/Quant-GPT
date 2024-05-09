#!/bin/bash

export MODELSCOPE_CACHE='/root/autodl-tmp'
export USE_MODELSCOPE_HUB=1
DATASET_NAME="ft_data_32k_train,ft_data_32k_test"
MODEL_NAME="v3"
MODEL_PATH="TongyiFinance/Tongyi-Finance-14B-Chat"
# MODEL_PATH="../exp_model/v2"

deepspeed --num_gpus 2 run_exp.py \
    --stage orpo \
    --do_train \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET_NAME \
    --dataset_dir ./ \
    --template 	qwen \
    --finetuning_type lora \
    --output_dir ../exp_model/$MODEL_NAME \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 32768 \
    --vllm_maxlen 32768 \
    --preprocessing_num_workers 32 \
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
    --val_size 0.1 \
    --shift_attn
    # --weight_decay 0.01 \
    # --max_samples 100 \
    # --print_param_status \
    # --max_grad_norm 1.0 \
    # --max_samples 10 \
    # --quantization_bit 4 \       
    # --streaming \
    # --max_steps 10000
    # --report_to tensorboard \
    # --quantization_bit 8 \
