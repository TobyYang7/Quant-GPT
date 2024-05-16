import pandas as pd
from datetime import datetime, timedelta


# TODO: 新闻类型筛选
def get_news(time_period, date_end = datetime.now(), file_path='A_news_train_last1000_withvector.feather'):
    df = pd.read_feather(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    data_start = date_end - timedelta(days=time_period)
    df_target_news = df[(df['datetime'] >= data_start) & (df['datetime'] <= date_end)].reset_index(drop=True)
    return df_target_news


if __name__ == "__main__":
    df = get_news(70)
    print(df)