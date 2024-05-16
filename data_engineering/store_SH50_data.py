#%%
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

def juejin_to_tushare(code: str):
    component = code.split('.')
    if component[0] == "SHSE":
        return component[1] + ".SH"
    elif component[0] == "SZSE":
        return component[1] + ".SZ"
    else:
        # this code should not be ran
        raise ValueError("should not have this case" + code)

SH50 = pd.read_csv("../data/info/000016SH.csv", index_col=0)
parquet_file = pq.ParquetFile('../data/announcement/LLM_data.parquet')
stock_prices = parquet_file.read().to_pandas()

stock_prices.symbol = stock_prices.symbol.apply(juejin_to_tushare)
filtered_stock_prices = stock_prices[stock_prices['symbol'].isin(SH50['con_code'].unique())]
# %%
table = pa.Table.from_pandas(filtered_stock_prices)
pq.write_table(table, '../data/announcement/LLM_data_SH50.parquet')
# %%
