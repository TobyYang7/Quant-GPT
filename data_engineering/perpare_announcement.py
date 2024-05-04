from data_engineering_tools import *
import tushare as ts

parquet_file = pq.ParquetFile('../data/announcement/LLM_data_SH50.parquet')
stock_prices = parquet_file.read().to_pandas()
stock_prices.bob = stock_prices.bob.apply(replace_default_datetime)
print(stock_prices)

with open("token.txt") as f:
    token = f.readline()

pro = ts.pro_api(token)

prepare_announcement(pro, stock_prices)