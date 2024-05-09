import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from modelscope import AutoTokenizer, snapshot_download

model_dir = snapshot_download('TongyiFinance/Tongyi-Finance-14B-Chat')
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

def calculate_token_length(text):
    global tokenizer
    tokens = tokenizer.encode(text, truncation=True)
    return len(tokens)

stock_list = ["600031.SH", "600036.SH", "600050.SH", "600104.SH", "600346.SH", "600570.SH", "600887.SH", "601390.SH", "603160.SH", "601668.SH"]
for stock in tqdm(stock_list):
    parquet_file = pq.ParquetFile('data/announcement/' + stock + '.parquet')
    news = parquet_file.read().to_pandas()
    content_token_len = []
    new_content = []
    for c in news.loc[:, "content"]:
        token = calculate_token_length(c)
        # control the token under 3.2 * 1e4 (32k)
        if token > 3.2 * 1e4:
            new_content.append(np.nan)
        else:
            new_content.append(c)
        content_token_len.append(token)
    news.loc[:, "content"] = new_content
    news.loc[:, "content_token_len"] = content_token_len
    news.dropna(inplace = True)
    table = pa.Table.from_pandas(news)
    pq.write_table(table, 'data/announcement/' + stock + '_filter_32k' + '.parquet')