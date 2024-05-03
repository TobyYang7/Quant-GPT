#!/bin/bash

export USE_MODELSCOPE_HUB=1
DATASET_NAME="ft_data_train"
MODEL_NAME="v1"
# MODEL_PATH="/home/zhangmin/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct"
MODEL_PATH="qwen/Qwen-7B-Chat"

deepspeed --num_gpus 4 run_exp.py \
    --stage sft \
    --do_train \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET_NAME \
    --dataset_dir ./ \
    --template 	qwen \
    --finetuning_type lora \
    --output_dir ./saves/$MODEL_NAME \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 8192 \
    --vllm_maxlen 8192 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_ratio 0.1 \
    --evaluation_strategy epoch \
    --learning_rate 5e-5 \
    --num_train_epochs 4 \
    --plot_loss \
    --fp16 \
    --load_best_model_at_end \
    --orpo_beta 0.05 \
    --save_strategy epoch \
    --do_eval false \
    --deepspeed  ds_z3_config.json \
    --ddp_timeout 180000000 \
    --max_samples 100 \
    # --print_param_status \
    # --max_grad_norm 1.0 \
    # --weight_decay 0.01 \
    # --max_samples 10 \
    # --quantization_bit 4 \       
    # --streaming \
    # --max_steps 10000
    # --report_to tensorboard \
    # --quantization_bit 8 \
