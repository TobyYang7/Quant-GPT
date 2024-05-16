from transformers.generation import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
from modelscope import GenerationConfig
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from tqdm import tqdm
import json
from llmtuner import ChatModel

# model_path = "/home/zhangmin/toby/Quant-GPT/exp_model/v4_cp2000"
# model_path = "/home/zhangmin/.cache/modelscope/hub/TongyiFinance/Tongyi-Finance-14B-Chat"
# model_path = "/home/zhangmin/.cache/modelscope/hub/ZhipuAI/chatglm3-6b-128k"
model_path = "/home/zhangmin/.cache/modelscope/hub/qwen/Qwen-7B-Chat"
model = ChatModel(dict(
    model_name_or_path=model_path,
    template="qwen",
    temperature=0.6,
    top_p=0.8,
    cutoff_len=8192,
    do_sample=True,
    max_new_tokens=256,
    adapter_name_or_path="/home/zhangmin/toby/Quant-GPT/train/saves/v6_cp200",
    # infer_backend="vllm",
    # vllm_gpu_util=0.9
))


# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
# inputs = tokenizer('蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是', return_tensors='pt')
# inputs = inputs.to(model.device)
# pred = model.generate(**inputs)
# print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))


# def predict(model, prompt):
#     inputs = tokenizer(prompt, return_tensors='pt')
#     inputs = inputs.to(model.device)

#     try:
#         with torch.no_grad():
#             pred = model.generate(**inputs, max_length=512)
#             response_text = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
#     except RuntimeError as exception:
#         if "out of memory" in str(exception):
#             print('WARNING: out of memory')
#             if hasattr(torch.cuda, 'empty_cache'):
#                 torch.cuda.empty_cache()
#             raise exception
#         else:
#             raise

#     return response_text

def predict(model, prompt):
    query = prompt
    messages = [{"role": "user", "content": query}]
    try:
        with torch.no_grad():
            resp = model.chat(messages, system="你是一个金融交易员")
            response_text = resp[0].response_text
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print('WARNING: out of memory')
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            raise exception
        else:
            raise

    return response_text


def evaluate_accuracy(model, test_data):
    results = []
    correct_predictions = 0
    total_predictions = 0
    output_file = 'test_dataset/v6_cp200.json'

    with tqdm(total=test_data.shape[0]) as pbar:
        for _, item in test_data.iterrows():
            text_input = f'''{item['prompt']}\n{item['content']}'''
            predicted_response = predict(model, text_input).replace(" ", "")

            expected_response = item['label']
            result = {
                "prompt": item['prompt'],
                "content": item['content'],
                "expected": expected_response,
                "predicted": predicted_response
            }
            results.append(result)
            total_predictions += 1
            if expected_response == predicted_response:
                correct_predictions += 1

            accuracy = correct_predictions / total_predictions
            pbar.set_description(f"{correct_predictions}/{total_predictions}, {predicted_response}, {expected_response}, {expected_response == predicted_response}")
            pbar.update()

            # 实时更新JSON文件
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    return accuracy


max_samples = 1400
test_data = pd.read_json('ft_data_summary_2k_test.json')
# test_data = test_data.sample(frac=1).reset_index(drop=True)
test_data = test_data[:max_samples]

accuracy = evaluate_accuracy(model, test_data)
print("Accuracy:", accuracy)
