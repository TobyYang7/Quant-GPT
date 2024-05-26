 export HF_HUB_ENABLE_HF_TRANSFER=1 
 export HF_ENDPOINT=https://hf-mirror.com
# export USE_MODELSCOPE_HUB=1

CUDA_VISIBLE_DEVICES=2,3 python web_demo.py \
    --template qwen \
    --model_name_or_path  bjdwh/UrbanGPT \
    --do_sample \
    --infer_backend vllm \
    # --flash_attn off 
    # --adapter_name_or_path /home/zhangmin/toby/Quant-GPT/exp_model/v3_adapter/checkpoint-600 \
