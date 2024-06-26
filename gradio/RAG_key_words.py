#%%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer

def retriever(model, query, df_news, score_weight=[0.7, 0.3], top_k=5):
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
        
def generater(query, document_list, llm_model):
    tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(llm_model, trust_remote_code=True, device_map='auto')
    model = model.eval()
    joined_doc = '\n\n\n'.join(document_list)
    prompt = f'The contents of reference articles are\n\n\n{joined_doc}\n\n\n Answer the following question according to the reference articles, If the reference articles do not mention the answer, return unkown. The question is:{query}'
    response, history = model.chat(tokenizer, prompt, history=[])
    return response

if __name__=="__main__":
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    # df_news_train = pd.read_feather('/home/zhangmin/toby/Quant-GPT/gradio/A_news_train_all_withvector.feather')
    df_news_test = pd.read_feather('/home/zhangmin/toby/Quant-GPT/gradio/A_news_test_all_withvector.feather')
    # df_news = pd.concat([df_news_train, df_news_test]).reset_index(drop=True)
    new_list = retriever(model, "建筑", df_news_test, score_weight=[0.7, 0.3], top_k=5)
# %%
