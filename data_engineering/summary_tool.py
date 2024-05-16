#%%
import time
import traceback
from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from GPT_tool.OpenAIGPT import *

def get_summary(stock_list, gpt):
    for stock in tqdm(stock_list):
        parquet_file = pq.ParquetFile('../data/announcement/' + stock + '.parquet')
        news = parquet_file.read().to_pandas()
        contents = news.content
        summaries = []
        for c in tqdm(contents):
            flag = True
            # whether we can go to next one
            while flag:
                try:
                    summary = gpt.call("请您帮忙总结下面的内容，尽可能不损失对投资有用的信息", c, "当然，下面是总结的信息：")
                    summaries.append(c.split("正文")[0] + '\n' + summary)
                    print(summary)
                    flag = False
                except openai.error.RateLimitError as e:
                    time.sleep(120)
                    print(e)             
                except: # openai.error.InvalidRequestError
                    summaries.append(c.split("正文")[0] + "\n由于正文过长，总结缺失。")
                    traceback.print_exc()
                    print(c)
                    flag = False
            
        news.summary = summaries
        table = pa.Table.from_pandas(news)
        pq.write_table(table, '../data/announcement/' + stock + '_summary' + '.parquet')

if '__main__' == __name__:
    unique_stocks_list = ["600031.SH", "600036.SH", "600050.SH", "600104.SH", "600346.SH", "600570.SH", "600887.SH", "601390.SH", "603160.SH", "601668.SH"]
    igpt = OpenAIGPT(keys_path="GPT_tool/gpt3keys.txt")
    get_summary(unique_stocks_list, igpt)
# %%
