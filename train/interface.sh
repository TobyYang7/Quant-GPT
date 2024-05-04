export USE_MODELSCOPE_HUB=1
MODEL_NAME="v1"

CUDA_VISIBLE_DEVICES=3 python web_demo.py \
    --template llama3 \
    --model_name_or_path qwen/Qwen-7B-Chat \
    --adapter_name_or_path saves/$MODEL_NAME \