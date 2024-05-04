#%%
import tushare as ts

with open("token.txt") as f:
    token = f.readline()

pro = ts.pro_api(token)
# %%
# import pyarrow as pa
# import pyarrow.parquet as pq
import time
import pandas as pd
from datetime import datetime, timedelta

today = '2018-10-08 09:00:00'
datetime_obj = datetime.strptime(today, "%Y-%m-%d %H:%M:%S")
end_date = '2024-04-14 09:00:00'

while today <= end_date:
    df = pro.news(src='sina', start_date=today, end_date=(datetime_obj+timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"), fields="datetime, content, title, channels")
    df.to_feather('../data/sina/{}.feather'.format(today[:10]))
    datetime_obj = datetime_obj+timedelta(days=1)
    today = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
    time.sleep(1)

#%%
import time
import pandas as pd
from datetime import datetime, timedelta

today = '2018-10-08 09:00:00'
datetime_obj = datetime.strptime(today, "%Y-%m-%d %H:%M:%S")
end_date = '2024-04-14 09:00:00'
df_selected = None
while today <= end_date:
    df = pd.read_feather('../data/sina/{}.feather'.format(today[:10]))
    channels = df["channels"]
    index_list = []
    for i in range(len(channels)):
        for D in channels[i]:
            if D["name"] == "A股":
                index_list.append(i)
    if df_selected is None:
        df_selected = df.loc[index_list]
    else:
        df_selected = pd.concat([df_selected, df.loc[index_list]], axis = 0)
    datetime_obj = datetime_obj+timedelta(days=1)
    today = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")

#%%
A_news = df_selected.reset_index(drop=True)
train_test_split = "2023-04-14 09:00:00"
A_news_train = A_news[A_news.loc[:, "datetime"] < train_test_split]
A_news_test = A_news[A_news.loc[:, "datetime"] >= train_test_split]

A_news_train.to_feather("../data/cleaned_news/A_news_train.feather")
A_news_test.to_feather("../data/cleaned_news/A_news_test.feather")
#%%
df = pro.major_news(src='', start_date='2022-01-01 09:00:00', end_date='2023-01-01 09:00:00', fields='title, content, pub_time')
#%%
df = pro.cctv_news(date='20230101')
# %%
import pandas as pd

A_news_labeled = pd.read_feather("../data/cleaned_data/A_news_labeled.feather")

