export USE_MODELSCOPE_HUB=1
MODEL_NAME="v1"

CUDA_VISIBLE_DEVICES=3 python web_demo.py \
    --template qwen \
    --model_name_or_path /home/zhangmin/.cache/modelscope/hub/TongyiFinance/Tongyi-Finance-14B-Chat \
    # --adapter_name_or_path saves/$MODEL_NAME \