import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer


def retriever(query, df_news, score_weight=[0.7, 0.3], top_k=5):
    dense_vecs = df_news['dense_vecs']
    lexical_weights = df_news['lexical_weights']               # ！！！！！！！！！！测试feather数据结构
    query_embedding = model.encode(query, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    score_list = []
    for i in range(len(dense_vecs)):
        score_dense = query_embedding['dense_vecs'] @ dense_vecs[i].T
        filtered_dict = {k: v for k, v in lexical_weights[i].items() if v is not None}
        score_lexical = model.compute_lexical_matching_score(query_embedding['lexical_weights'], filtered_dict)
        score_list.append(score_dense * score_weight[0] + score_lexical * score_weight[1])

    sorted_lst_with_idx = sorted(enumerate(score_list), key=lambda x: x[1], reverse=True)
    top_rank_score_and_idxs = sorted_lst_with_idx[:top_k]
    print('The target chunks idxs and scores') 
    print(top_rank_score_and_idxs)
    news_list = []
    for i in top_rank_score_and_idxs:
        news_time = df_news.loc[i[0]]['datetime']
        news_content = df_news.loc[i[0]]['content']
        news_list.append(f'{news_time} 新闻:{news_content}')

    return news_list
        
def retriever_2(query, embedding_model, df_news, score_weight=[0.7, 0.3], top_k=5):
    model = BGEM3FlagModel(embedding_model, use_fp16=True)
    dense_vecs = df_news['dense_vecs']             # ！！！！！！！！！！测试feather数据结构
    query_embedding = model.encode(query, return_dense=True, return_sparse=False, return_colbert_vecs=False)
    score_list = []
    for i in range(len(dense_vecs)):
        score_dense = query_embedding['dense_vecs'] @ dense_vecs[i].T
        score_list.append(score_dense)

    sorted_lst_with_idx = sorted(enumerate(score_list), key=lambda x: x[1], reverse=True)
    top_rank_score_and_idxs = sorted_lst_with_idx[:top_k]
    print('The target chunks idxs and scores') 
    print(top_rank_score_and_idxs)
    news_list = []
    for i in top_rank_score_and_idxs:
        news_time = df_news.loc[i[0]]['datetime']
        news_content = df_news.loc[i[0]]['content']
        news_list.append(f'{news_time} 新闻:{news_content}')

    return news_list

def generater(query, document_list, llm_model):
    tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(llm_model, trust_remote_code=True, device_map='auto')
    model = model.eval()
    joined_doc = '\n\n\n'.join(document_list)
    prompt = f'The contents of reference articles are\n\n\n{joined_doc}\n\n\n Answer the following question according to the reference articles, If the reference articles do not mention the answer, return unkown. The question is:{query}'
    response, history = model.chat(tokenizer, prompt, history=[])
    return response
        
        
if __name__=="__main__":
    llm_model = "/home/zhangmin/.cache/modelscope/hub/qwen/Qwen-7B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(llm_model,  trust_remote_code=True)
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    df_news_train = pd.read_feather('/home/zhangmin/toby/Quant-GPT/gradio/A_news_train_all_withvector.feather')
    df_news_test = pd.read_feather('/home/zhangmin/toby/Quant-GPT/gradio/A_news_test_all_withvector.feather')
    df_news = pd.concat([df_news_train, df_news_test]).reset_index(drop=True)
    # df_news = pd.read_feather('/home/zhangmin/toby/Quant-GPT/gradio/A_news_train_last1000_withvector.feather')
    df_news['datetime'] = pd.to_datetime(df_news['datetime'])
    print(df_news)
    print('data load successfully')
    for id in ['600031.SH', 
               '600036.SH', '600050.SH', '600104.SH', '600346.SH', '600570.SH', '600887.SH', '601390.SH', '601668.SH', '603160.SH'
               ]:
        all_news_list = []
        all_num_token = []
        df_summary = pd.read_parquet(f'/home/zhangmin/toby/Quant-GPT/data/announcement/{id}_summary_new.parquet')
        for index, row in df_summary.iterrows():
            date_end = datetime.strptime(row['rec_time'], "%Y-%m-%d %H:%M:%S")
            data_start = date_end - timedelta(days=90)
            print(date_end)
            print(data_start)
            df_target_news = df_news[(df_news['datetime'] >= data_start) & (df_news['datetime'] <= date_end)].reset_index(drop=True)
            new_list = retriever(row['summary'], df_target_news, score_weight=[0.7, 0.3], top_k=5)
            print(new_list)
            joint_news = '\n'.join(new_list)
            tokens = tokenizer.encode(joint_news, truncation=True)
            all_news_list.append(joint_news)
            all_num_token.append(len(tokens))
        df_summary.loc[:, 'macro'] = all_news_list
        df_summary.loc[:, 'macro_token_len'] = all_num_token
        df_summary.to_parquet(f'/home/zhangmin/toby/Quant-GPT/data/announcement/{id}_summary_with_macro.parquet', engine='pyarrow')
        print(id)
    # print(df_summary.head())
    # query = "what is IDE"
    # document_list = retriever(query, embedding_model, df_news, top_k=1)
    # response = generater(query,document_list, llm_model)
    # print(response)
    