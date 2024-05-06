import time
from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from llmtuner import ChatModel

model_path = "/home/zhangmin/.cache/modelscope/hub/TongyiFinance/Tongyi-Finance-14B-Chat"
# model_path = "../exp_model/v2_14b"

model = ChatModel(dict(
    model_name_or_path=model_path,
    template="qwen",
    temperature=0.5,
    cutoff_len=8192,
    infer_backend="vllm",
    max_new_tokens=1024,
    vllm_gpu_util = 0.7
))

def predict(model, prompt):
    query = prompt
    messages = [{"role": "user", "content": query}]
    resp = model.chat(messages, system="请您帮忙总结下面的内容，尽可能不损失对投资有用的信息。")
    return (resp[0].response_text)

def calculate_token_length(text):
    model_name = "/home/zhangmin/.cache/modelscope/hub/qwen/Qwen-7B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name,  trust_remote_code=True)
    tokens = tokenizer.encode(text, truncation=True)
    return len(tokens)

def summary(model, content):
    content_token = calculate_token_length(content)
    if content_token >= 2048:
        # binary split and overlap in the central text
        split_content = content.split('\n')
        if content_token < 1e4:
            summary1 = summary(model, "请您帮忙总结下面的切分内容（部分内容可能截断），尽可能不损失对投资有用的信息" + ('\n').join(split_content[:int(len(split_content)/9*5)]))
            summary2 = summary(model, "请您帮忙总结下面的切分内容（部分内容可能截断），尽可能不损失对投资有用的信息" + ('\n').join(split_content[int(len(split_content)/9*4):]))
        else:
            summary1 = summary(model, "请您帮忙总结下面的切分内容（部分内容可能截断），尽可能不损失对投资有用的信息" + ('\n').join(split_content[:int(len(split_content)/15*8)]))
            summary2 = summary(model, "请您帮忙总结下面的切分内容（部分内容可能截断），尽可能不损失对投资有用的信息" + ('\n').join(split_content[int(len(split_content)/15*7):]))
        print(summary1, '\n', summary2)
        final_summary = summary(model, "请您帮忙总结下面拼接的内容（中间部分内容可能重叠），尽可能不损失对投资有用的信息" + summary1 + '\n' +  summary2)
        print(final_summary)
        return final_summary
    else:
        return predict(model, "请您帮忙总结下面的内容，尽可能不损失对投资有用的信息。\n" + content)
    
stock_list = ["600031.SH", "600036.SH", "600050.SH", "600104.SH", "600346.SH", "600570.SH", "600887.SH", "601390.SH", "603160.SH", "601668.SH"]
for stock in tqdm(stock_list):
    parquet_file = pq.ParquetFile('data/announcement/' + stock + '.parquet')
    news = parquet_file.read().to_pandas()
    contents = news.content
    summaries = []
    summary_token_lens = []
    content_token_lens = []
    for c in tqdm(contents):
        print(c.split("正文")[0])
        content_token_lens.append(calculate_token_length(c))
        summaries.append(c.split("正文")[0] + '\n' + summary(model, c))
        summary_token_lens.append(calculate_token_length(summaries[-1]))
        
    news.content_token_len = content_token_lens
    news.summary = summaries
    news.summary_token_len = summary_token_lens
    table = pa.Table.from_pandas(news)
    pq.write_table(table, 'data/announcement/' + stock + '_summary' + '.parquet')