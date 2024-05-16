import pandas as pd
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor
import torch

from llmtuner import ChatModel

# model_path = "/home/zhangmin/.cache/modelscope/hub/TongyiFinance/Tongyi-Finance-14B-Chat-llamafy"
# adapter = "/home/zhangmin/toby/Quant-GPT/exp_model/v3_adapter/checkpoint-600"
model_path = "../exp_model/v2_14b"

model = ChatModel(dict(
    model_name_or_path=model_path,
    template="qwen",
    temperature=0.5,
    cutoff_len=8192,
    infer_backend="vllm",
    max_new_tokens=1024,
    # adapter_name_or_path=adapter
))

# def chat(user_input):
#     model_path = "../exp_model/v1"
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)
#     messages = [
#         {"role": "system", "content": ''},
#         {"role": "user", "content": user_input}
#     ]

#     input_ids = tokenizer.apply_chat_template(
#         messages, add_generation_prompt=True, return_tensors="pt"
#     ).to(model.device)

#     outputs = model.generate(
#         input_ids,
#         max_new_tokens=1024,
#         do_sample=True,
#         temperature=0.5,
#         top_p=0.5,
#     )
#     response = outputs[0][input_ids.shape[-1]:]
#     return tokenizer.decode(response, skip_special_tokens=True)

# model_path = "../exp_model/v1"
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)


def process_row(row):
    try:
        stock_mapping = {
            "600031.SH": "三一重工",
            "600036.SH": "招商银行",
            "600050.SH": "中国联通",
            "600104.SH": "上汽集团",
            "600346.SH": "恒力石化",
            "600570.SH": "恒生电子",
            "600887.SH": "伊利股份",
            "601390.SH": "中国中铁",
            "603160.SH": "汇项科技",
            "601668.SH": "中国建筑"
        }
        stock = stock_mapping[row['symbol']]
        prompt = f"请根据以下新闻文本，预测{stock}股票的对数收益率属于以下哪一类别 (极度负面/负面/中性/正面/极度正面)"
        label_mapping = {0: "极度负面", 1: "负面", 2: "中性", 3: "正面", 4: "极度正面"}
        label = label_mapping[row['label']]

        def chat(user_input):
            global model
            query = prompt
            messages = [{"role": "user", "content": query}]
            resp = model.chat(messages, system="你是一个金融交易员")
            return (resp[0].response_text)

        predict = chat(prompt+row['content'])
        torch.cuda.empty_cache()
        return {
            "prompt": prompt,
            "content": row['content'],
            "label": label,
            "predict": predict
        }
    except Exception as e:
        print(f"Error processing row: {e}")
        return None


def generate_dataset(df, max_samples=32):
    df_subset = df.head(max_samples)
    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_row, row) for _, row in df_subset.iterrows()]
        for future in tqdm(futures, total=len(futures), desc="Processing rows"):
            result = future.result()
            if result:
                results.append(result)

    return results


data = pd.read_json('ft_data_summary.json')
dataset = generate_dataset(data, max_samples=10000)
output_file_path = 'results.jsonl'
with open(output_file_path, 'w', encoding='utf-8') as f:
    for item in dataset:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

cnt = 0
for line in dataset:
    if line['label'] in line['predict']:
        cnt += 1
print(f"Accuracy: {cnt/len(dataset)}")
