export USE_MODELSCOPE_HUB=1
MODEL_NAME="v2_14b"

CUDA_VISIBLE_DEVICES=0,1,2,3 python cli_demo.py \
    --template qwen \
    --model_name_or_path TongyiFinance/Tongyi-Finance-14B-Chat \
    --adapter_name_or_path saves/$MODEL_NAME \
    # --infer_backend vllm