import pandas as pd
from datetime import datetime, timedelta


# TODO: 新闻类型筛选
def get_news(time_period, file_path='A_news_train.feather'):
    df = pd.read_feather(file_path)
    date_now = datetime.now()

    data_start = date_now - timedelta(days=time_period)
    data_format = "%Y-%m-%d %H:%M:%S"
    Index_end = len(df['datetime'])
    for i in range(len(df['datetime'])):
        if datetime.strptime(df['datetime'][i], data_format) < data_start:
            continue
        else:
            Index_end = i
            break
    return df[Index_end:].reset_index()


if __name__ == "__main__":
    df = get_news(70)
    print(df)
