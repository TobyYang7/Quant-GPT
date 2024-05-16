#%%
import tushare as ts

with open("token.txt") as f:
    token = f.readline()

pro = ts.pro_api(token)

#%%
import numpy as np
import pandas as pd

trade_calendar = pro.trade_cal(exchange='', start_date='20181008', end_date='20240414')
# trade_calendar.to_csv("data/info/calendar.csv")

#%%
# 上证50
she50_weight = pro.index_weight(index_code='000016.SH', start_date='20181008', end_date='20240414')

# 科创50
kc50_weight = pro.index_weight(index_code='000688.SH', start_date='20181008', end_date='20240414')

she50_weight.to_csv("../data/info/000016SH.csv")
kc50_weight.to_csv("../data/info/000688SH.csv")

#%%
"""
stock_codes = list(set(np.concatenate([she50_weight.con_code.unique(), kc50_weight.con_code.unique()])))
stock_prices = None
for stock in stock_codes:
    if isinstance(stock_prices, pd.DataFrame):
        stock_prices = pd.concat([stock_prices, ts.pro_bar(ts_code=stock, adj='hfq', start_date='20181008', end_date='20240414')], axis = 0)
    else:
        stock_prices = ts.pro_bar(ts_code=stock, adj='hfq', start_date='20181008', end_date='20240414')

stock_prices = stock_prices.reset_index(drop=True)
stock_prices.to_csv("data/announcement/stocks_price.csv")
"""