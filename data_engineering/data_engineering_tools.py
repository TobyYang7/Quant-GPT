# %%
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
from keywords import *

#%%
def cal_log_return(data: pd.DataFrame, k: int):
    return np.log(data.shift(-k)/data)


def juejin_to_tushare(code: str):
    component = code.split('.')
    if component[0] == "SHSE":
        return component[1] + ".SH"
    elif component[0] == "SZSE":
        return component[1] + ".SZ"
    else:
        # this code should not be ran
        raise ValueError("should not have this case" + code)

def tushare_to_juejin(code: str):
    component = code.split('.')
    if component[1] == "SH":
        return "SHSE." + component[0]
    elif component[1] == "SZ":
        return "SZSE." + component[0]
    else:
        # this code should not be ran
        raise ValueError("should not have this case" + code)

def replace_default_datetime(time: datetime):
    return time.replace(hour=9)


def prepare_k_days_return(stock_prices: pd.DataFrame, k: int):
    stock_prices.reset_index(inplace = True, drop = True)
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
        return pd.concat([new1, new2], axis=0).reset_index(drop=True)
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


# def binary_search(arr, target):
#     target = pd.Timestamp(target)
#     arr = pd.to_datetime(arr)
#     # Calculate the absolute differences
#     abs_diffs = np.abs(arr - target)
#     # Find the index of the smallest difference
#     closest_index = abs_diffs.argmin()
#     return closest_index


def prepare_k_items_announcement_with_time_restriction(stock_prices: pd.DataFrame, unique_stocks_list: list, k: int, len_reference_list, reference_list, t_limit: int = None, token_limit: int = 3.2*1e4, postfix: str = None, with_macro=False):
    combined_info = []
    stock_symbols = []
    log_rs = []
    trading_time = []
    model_dir = "/home/zhangmin/.cache/modelscope/hub/ZhipuAI/chatglm3-6b-128k"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    def calculate_token_length(text):
        nonlocal tokenizer
        tokens = tokenizer.encode(text, truncation=True)
        return len(tokens)

    for stock in tqdm(unique_stocks_list):
        parquet_file = pq.ParquetFile('../data/announcement/' + stock + postfix + '.parquet')
        news = parquet_file.read().to_pandas()
        stock_price = stock_prices[stock_prices["symbol"] == stock]
        stock_price = stock_price.sort_values("bob").reset_index(drop=True)
        news = news[pd.to_datetime(news.loc[:, "rec_time"]) >= stock_price.loc[:, "bob"][0].to_datetime64()]
        news = news.sort_values("rec_time").reset_index(drop=True)

        working_list = []
        news_times = []
        token_len = []

        j = 0  # i is the pointer for stock_price, j is the pointer for stock_price
        while j < len(news):
            print(j, len(news))
            news_time = datetime.strptime(news.loc[j, 'rec_time'], '%Y-%m-%d %H:%M:%S')
            # binary search find the stock return data after the news
            
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
            if with_macro:
                new_added = news.loc[:, len_reference_list][j] + news.loc[:, "macro_token_len"][j]
            else:
                new_added = news.loc[:, len_reference_list][j]
            while sum(token_len) > token_limit - new_added - news.loc[:, "industry_token_len"][j]:
                if len(news_times) == 0:
                    print(sum(token_len))
                    Warning("The original data token limit exceeded")
                    break
                working_list.pop(0)
                news_times.pop(0)
                token_len.pop(0)
            
            if with_macro:
                token_len.append(news.loc[:, len_reference_list][j] + news.loc[:, "macro_token_len"][j])
                working_list.append(news.loc[:, "macro"][j] + '\n\n\n' + news.loc[:, reference_list][j] + '\n\n\n')
            else:
                token_len.append(news.loc[:, len_reference_list][j])
                working_list.append(news.loc[:, reference_list][j] + '\n')
            news_times.append(news_time)
            combined_info.append(f"现在是{stock_price.bob[i]}\n" + stock_price.loc[:, "previous_index_price"][i] + "\n\n" + news.loc[:, "industry"][j] + "\n\n" +  "\n".join(working_list))
            assert (calculate_token_length(combined_info[-1]) <= token_limit + 160)
            stock_symbols.append(stock)
            log_rs.append(stock_price.log_r[i])
            trading_time.append(return_time)

            j += 1

    return pd.DataFrame({"symbol": stock_symbols, "time": trading_time, "content": combined_info, "log_r": log_rs})


def get_previous_index_price(time_str):
    parquet_file = pq.ParquetFile('../data/index_info/SH_index_data.parquet')
    index_price = parquet_file.read().to_pandas()
    end_index = index_price[index_price.loc[:, "bob"] == time_str].index[0]
    return "前五个交易日上证指数的价格如下" + ",".join(map(str, list(index_price.iloc[end_index - 6:end_index - 1].loc[:, "open"])))

# %%
if __name__ == '__main__':
    # table = pa.Table.from_pandas(df)
    # pq.write_table(table, 'output.parquet')
    parquet_file = pq.ParquetFile('../data/announcement/LLM_data_SH50.parquet')
    stock_prices = parquet_file.read().to_pandas()
    unique_stocks_list = ["600031.SH", "600036.SH", "600050.SH", "600104.SH", "600346.SH", "600570.SH", "600887.SH", "601390.SH", "603160.SH", "601668.SH"]
    # stock_prices.loc[:, "symbol"] = stock_prices.loc[:, "symbol"].apply(juejin_to_tushare)
    stock_prices = stock_prices[stock_prices.loc[:, "symbol"].isin(unique_stocks_list)]
    # If your data is daily, run below code
    stock_prices.loc[:, "previous_index_price"] = stock_prices.bob.apply(get_previous_index_price)
    stock_prices.loc[:, "bob"] = stock_prices.bob.apply(replace_default_datetime)
    log_r = prepare_k_days_return(stock_prices, 5)
    print(log_r)
    # unique_stocks_list = ["600031.SH", "600036.SH", "600050.SH"]
    data = prepare_k_items_announcement_with_time_restriction(log_r, unique_stocks_list, 5, "summary_token_len", "summary", 10, 8*1e3, 'summary_with_industry', False)
    table = pa.Table.from_pandas(data)
    pq.write_table(table, '../data/announcement/processed_LLM_data_5_5_10_8k_summary_with_industry.parquet')
    
    """
    files = os.listdir("../data/industry_data")
    llm_model = "/home/zhangmin/.cache/modelscope/hub/qwen/Qwen-7B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(llm_model,  trust_remote_code=True)
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    df_news_train = pd.read_feather('/home/zhangmin/toby/Quant-GPT/gradio/A_news_train_all_withvector.feather')
    df_news_test = pd.read_feather('/home/zhangmin/toby/Quant-GPT/gradio/A_news_test_all_withvector.feather')
    df_news = pd.concat([df_news_train, df_news_test]).reset_index(drop=True)
    df_news['datetime'] = pd.to_datetime(df_news['datetime'])
    
    selected_data = None
    
    for file in tqdm(files):
        trade_data = pd.read_csv("../data/industry_data/" + file, index_col=0)
        industry = file[:-4]
        if "其他" in industry:
            continue
        if len(trade_data) == 0:
            continue
        trade_data = trade_data.sort_values("trade_date").reset_index(drop=True)
        if "len_token" in trade_data.columns:
            trade_data.loc[:, "log_r"] = cal_log_return(trade_data.loc[:, "open"], 3)
            trade_data = trade_data[trade_data.loc[:, "len_token"] > 0]
            trade_data.loc[:, "industry"] = [industry for _ in range(len(trade_data))]
            if selected_data is None:
                selected_data = trade_data
            else:
                selected_data = pd.concat([selected_data, trade_data])
            continue
        trade_data['trade_date'] = pd.to_datetime(trade_data['trade_date'], format="%Y%m%d")
        time_delta = timedelta(hours=9)
        trade_data.loc[:, "trade_date"] = trade_data.loc[:, "trade_date"] + time_delta
        industry_news = []
        len_token = []
        for index, row in trade_data.iterrows():
            date_end = row["trade_date"]
            data_start = date_end - timedelta(days=1)
            df_target_news = df_news[(df_news['datetime'] >= data_start) & (df_news['datetime'] <= date_end)].reset_index(drop=True)
            print(index, len(trade_data))
            accepted_news = []
            new_list, top_rank_score_and_idxs = retriever(model, industry, df_target_news, score_weight=[0.7, 0.3], top_k=5)
            for index, item in enumerate(top_rank_score_and_idxs):
                if item[1] > 0.4:
                    accepted_news.append(new_list[index])
            if len(accepted_news) == 0:
                industry_news.append("")
                len_token.append(0)
            else:
                industry_news.append("\n".join(accepted_news))
                tokens = tokenizer.encode(industry_news[-1], truncation=True)
                print(industry_news[-1])
                len_token.append(len(tokens))
        trade_data.loc[:, "industry_news"] = industry_news
        trade_data.loc[:, "len_token"] = len_token
        
        trade_data.to_csv("../data/industry_data_add_news/" + file)
        
        trade_data.loc[:, "log_r"] = cal_log_return(trade_data.loc[:, "open"], 3)
        trade_data = trade_data[trade_data.loc[:, "len_token"] > 0]
        trade_data.loc[:, "industry"] = [industry for _ in range(len(trade_data))]
        if selected_data is None:
            selected_data = trade_data
        else:
            selected_data = pd.concat([selected_data, trade_data])
    
    selected_data.to_csv("..data/selected_data.csv")
# %%
    files = os.listdir("../data/industry_data_add_news")
    selected_data = None
    for file in tqdm(files):
        trade_data = pd.read_csv("../data/industry_data_add_news/" + file, index_col=0)
        industry = file[:-4]
        trade_data = trade_data.sort_values("trade_date").reset_index(drop=True)
        if "len_token" in trade_data.columns:
            trade_data.loc[:, "log_r"] = cal_log_return(trade_data.loc[:, "open"], 3)
            trade_data = trade_data[trade_data.loc[:, "len_token"] > 0]
            trade_data.loc[:, "industry"] = [industry for _ in range(len(trade_data))]
            if selected_data is None:
                selected_data = trade_data
            else:
                selected_data = pd.concat([selected_data, trade_data])
    
    selected_data.loc[:, "content"] = "现在是" + selected_data.loc[:, "trade_date"] + "\n" + selected_data.loc[:, "industry_news"]
    
    selected_data.to_csv("../data/industry_data.csv")
    """
# %%
