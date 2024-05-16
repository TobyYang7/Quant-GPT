from modelscope import GenerationConfig
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
import time
from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
# from transformers import AutoTokenizer
from llmtuner import ChatModel
import os

# export CUDA_VISIBLE_DEVICES=1,2,3
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
print(os.environ.get("CUDA_VISIBLE_DEVICES"))


model_path = "/home/zhangmin/.cache/modelscope/hub/ZhipuAI/chatglm3-6b-128k"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = ChatModel(dict(
    model_name_or_path=model_path,
    template="chatglm3",
    temperature=0.5,
    cutoff_len=1024*64,
    max_new_tokens=1024*2,
    do_sample=True
    # infer_backend="vllm",
    # vllm_gpu_util=0.7,
))


def predict(model, prompt):
    query = prompt
    messages = [{"role": "user", "content": query}]
    resp = model.chat(messages, system="请您帮忙总结下面的内容，尽可能不损失对投资有用的信息。")
    return (resp[0].response_text)


def calculate_token_length(text):
    global tokenizer
    tokens = tokenizer.encode(text, truncation=True)
    return len(tokens)


def summary(model, content):
    # print('Start summary')
    original_token_count = calculate_token_length(content)
    print("原始Token数量:", original_token_count)
    if original_token_count >= 70 * 1024:
        mid_point = len(content) // 2
        while content[mid_point] != ' ' and mid_point < len(content):
            mid_point += 1
        part1 = summary(model, content[:mid_point])
        part2 = summary(model, content[mid_point:])
        res = part1 + ' ' + part2
        print("Token数量:", calculate_token_length(res))
        print(res)
        return res
    else:
        res = predict(model, "请您帮忙总结下面的内容，尽可能不损失对投资有用的信息。\n" + content)
        print("Token数量:", calculate_token_length(res))
        print(res)
        return res


# pid
print(f"pid:{os.getpid()}")
# stock_list = ["600031.SH", "600036.SH", "600050.SH", "600104.SH", "600346.SH", "600570.SH", "600887.SH", "601390.SH", "603160.SH", "601668.SH"]
stock_list = ["600570.SH", "600887.SH", "601390.SH", "603160.SH", "601668.SH"]
for stock in tqdm(stock_list):
    parquet_file = pq.ParquetFile('../data/announcement/' + stock + '.parquet')
    news = parquet_file.read().to_pandas()
    contents = news.content
    summaries = []
    summary_token_lens = []
    content_token_lens = []
    for c in tqdm(contents):
        # print(c.split("正文")[0])
        content_token_lens.append(calculate_token_length(c))
        summaries.append(c.split("正文")[0] + '\n' + summary(model, c))
        summary_token_lens.append(calculate_token_length(summaries[-1]))
    news.loc[:, "content_token_len"] = content_token_lens
    news.loc[:, "summary"] = summaries
    news.loc[:, "summary_token_len"] = summary_token_lens
    table = pa.Table.from_pandas(news)
    pq.write_table(table, '../data/announcement/' + stock + '_summary_new' + '.parquet')
