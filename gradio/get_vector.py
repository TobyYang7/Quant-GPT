import pandas as pd
import numpy as np
import json
import os
from FlagEmbedding import BGEM3FlagModel

def write_vecs(df_news, embedding_model_path):
    model = BGEM3FlagModel(embedding_model_path,  use_fp16=True)
# 逐行添加新列
    df_news['dense_vecs'] = pd.Series(dtype='object')
    df_news['lexical_weights'] = pd.Series(dtype='object')
    for idx in df_news.index:
        print(df_news.loc[idx, 'content'])
        embedding_result = model.encode(df_news.at[idx, 'content'], return_dense=True, return_sparse=True, return_colbert_vecs=False)
        print(embedding_result['dense_vecs'])
        print(embedding_result['lexical_weights'])
        df_news.at[idx, 'dense_vecs'] = embedding_result['dense_vecs']
        df_news.at[idx, 'lexical_weights'] = embedding_result['lexical_weights']
    df_news.to_feather('A_news_test_all_withvector_.feather')
    return df_news
        


if __name__ == "__main__":
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    embedding_model_path = 'BAAI/bge-m3'
    # df_news = pd.read_feather('/home/zhangmin/toby/Quant-GPT/data/cleaned_news copy/A_news_test.feather')
    # df_news = df_news[:].reset_index(drop=True)
    # df_news = write_vecs(df_news, embedding_model_path)
    # print(df_news)
    df_news = pd.read_feather('/home/zhangmin/toby/Quant-GPT/gradio/A_news_test_all_withvector.feather')
    print(df_news)
