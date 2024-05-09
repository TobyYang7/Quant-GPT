#%%
import os
import threading
import pdfplumber
import requests
from io import BytesIO
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta
from modelscope import AutoTokenizer, snapshot_download

#%%
def cal_log_return(data: pd.DataFrame, k: int):
    return np.log(data/data.shift(k))

def juejin_to_tushare(code: str):
    component = code.split('.')
    if component[0] == "SHSE":
        return component[1] + ".SH"
    elif component[0] == "SZSE":
        return component[1] + ".SZ"
    else:
        # this code should not be ran
        raise ValueError("should not have this case" + code)
    
def replace_default_datetime(time: datetime):
    return time.replace(hour = 9)

def prepare_k_days_return(stock_prices: pd.DataFrame, k: int):
    stock_prices["log_r"] = stock_prices.groupby(["symbol"])["open"].apply(cal_log_return, k).droplevel(0)
    return stock_prices

def extract_data_with_limit(pro, ts_code: str, start_date='20181008', end_date='20240414'):
    news = pro.anns_d(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=["ann_date", "ts_code", "name", "title", "url", "rec_time"])
    # there is a limit in the data extraction of pro
    if len(news) == 2000: 
        low = "20181008"
        high = "20240414"
        mid_1 = str((int(low[:4]) + int(high[:4]))//2) + "0101"
        mid_2 = str((int(low[:4]) + int(high[:4]))//2) + "0102"
        new1 = extract_data_with_limit(pro, ts_code, low, mid_1)
        new2 = extract_data_with_limit(pro, ts_code, mid_2, high)
        return pd.concat([new1, new2], axis = 0).reset_index(drop=True)
    else:
        return news

def process_data(data, start, end, result):
    content_list = []
    for i in tqdm(range(start, end)):
        row = data.iloc[i]
        url = row.url
        title = row.title
        time = row.rec_time
        response = requests.get(url)
        pdf_file = BytesIO(response.content)
        with pdfplumber.open(pdf_file) as pdf:
            pdf_text = ""
            for page in pdf.pages:
                pdf_text += page.extract_text()
        content_list.append("时间：" + time + '\n' + "标题：" + title + '\n' + "正文：" + pdf_text)
    result.extend(content_list)

def refine_news_item(data):
    content_list = []
    num_threads = 15  # define the number of threads

    threads = []
    chunk_size = len(data) // num_threads
    results = [[] for _ in range(num_threads)]

    for i in range(num_threads):
        start = i * chunk_size
        end = start + chunk_size if i < num_threads - 1 else len(data)
        t = threading.Thread(target=process_data, args=(data, start, end, results[i]))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    for res in results:
        content_list.extend(res)

    return content_list

def prepare_announcement(pro, stock_prices: pd.DataFrame):
    # stock_prices.symbol = stock_prices.symbol.apply(juejin_to_tushare)
    unique_stocks_list = stock_prices.symbol.unique()
    for stock in unique_stocks_list:
        # whether we have done it before
        if os.path.exists("../data/announcement/" + stock + ".parquet"):
            continue
        print(stock)
        news = extract_data_with_limit(pro, stock)
        news.loc[:, "content"] = refine_news_item(news)
        table = pa.Table.from_pandas(news)
        pq.write_table(table, f'../data/announcement/{stock}.parquet')

def binary_search(arr, target):
    """Get the smallest value greater than target"""
    left, right = 0, len(arr) - 1
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return right

def prepare_k_items_announcement_with_time_restriction(stock_prices: pd.DataFrame, unique_stocks_list: list, k: int, t_limit: int = None, token_limit: int = 3.2*1e4):
    combined_info = []
    stock_symbols = []
    log_rs = []
    trading_time = []
    model_dir = snapshot_download('TongyiFinance/Tongyi-Finance-14B-Chat')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    def calculate_token_length(text):
        nonlocal tokenizer
        tokens = tokenizer.encode(text, truncation=True)
        return len(tokens)

    for stock in tqdm(unique_stocks_list):
        parquet_file = pq.ParquetFile('../data/announcement/' + stock + '_filter_32k.parquet')
        news = parquet_file.read().to_pandas()
        news = news.sort_values("rec_time").reset_index(drop=True)

        stock_price = stock_prices[stock_prices["symbol"] == stock]
        stock_price = stock_price.sort_values("bob").reset_index(drop=True)

        working_list = []
        news_times = []
        token_len = []
        
        j = 0 # i is the pointer for stock_price, j is the pointer for stock_price
        while j < len(news):
            news_time = datetime.strptime(news.loc[j, 'rec_time'], '%Y-%m-%d %H:%M:%S')
            i = binary_search([stock_price.loc[idx, 'bob'].replace(tzinfo=None) for idx in range(len(stock_price))], news_time)

            return_time = stock_price.bob[i].replace(tzinfo=None)
            news_time = datetime.strptime(news.rec_time[j], '%Y-%m-%d %H:%M:%S')
            
            if t_limit:
                if len(news_times) > 0:
                    while news_times[0] < return_time - timedelta(days=t_limit):
                        working_list.pop(0)
                        news_times.pop(0)
                        token_len.pop(0)
                        if len(news_times) == 0:
                            break
            if len(working_list) >= k:
                working_list.pop(0)
                news_times.pop(0)
                token_len.pop(0)
            while sum(token_len) > token_limit - news.content_token_len[j]:
                working_list.pop(0)
                news_times.pop(0)
                token_len.pop(0)
                if len(news_times) == 0:
                    Warning("The original data token limit exceeded")
                    break
                
            token_len.append(news.content_token_len[j])
            working_list.append(news.content[j])
            news_times.append(news_time)
            combined_info.append(f"现在是{stock_price.bob[i]}\n" + "\n".join(working_list))
            assert(calculate_token_length(combined_info[-1]) < token_limit)
            stock_symbols.append(stock)
            log_rs.append(stock_price.log_r[i])
            trading_time.append(return_time)

            j += 1

    return pd.DataFrame({"symbol": stock_symbols, "time": trading_time, "content": combined_info, "log_r": log_rs})
#%%
if __name__ == '__main__':
    # table = pa.Table.from_pandas(df)
    # pq.write_table(table, 'output.parquet')

    parquet_file = pq.ParquetFile('../data/announcement/LLM_data_SH50.parquet')
    stock_prices = parquet_file.read().to_pandas()
    stock_prices.bob = stock_prices.bob.apply(replace_default_datetime)
    print(stock_prices)
    log_r = prepare_k_days_return(stock_prices, 1)
    unique_stocks_list = ["600031.SH", "600036.SH", "600050.SH", "600104.SH", "600346.SH", "600570.SH", "600887.SH", "601390.SH", "603160.SH", "601668.SH"]
    data = prepare_k_items_announcement_with_time_restriction(log_r, unique_stocks_list, 10, 30, 3.2*1e4)
# %%
    table = pa.Table.from_pandas(data)
    pq.write_table(table, '../data/announcement/processed_LLM_data_1_10_30_32k.parquet')
# %%
