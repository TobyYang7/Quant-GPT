#%%
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
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

    return news_list, top_rank_score_and_idxs
        
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
    df_news['datetime'] = pd.to_datetime(df_news['datetime'])
    unique_stocks_list = ["600031.SH", "600036.SH", "600050.SH", "600104.SH", "600346.SH", "600570.SH", "600887.SH", "601390.SH", "603160.SH", "601668.SH"]
    postfix = "_summary_new"
    for stock in tqdm(unique_stocks_list):
        parquet_file = pq.ParquetFile('../data/announcement/' + stock + postfix + '.parquet')
        news_summary = parquet_file.read().to_pandas()
        stock_keywords = {"600031.SH": ["大盘", "大市值", "工程机械", "工业互联网", "军民融合", "新能源车", "工业", "一带一路", "军工"],
                      "600036.SH": ["大盘", "大市值", "央企改革", "跨境支付", "国企改革", "互联金融", "深圳特区"],
                      "600050.SH": ["大盘", "大市值", "央企改革", "数据要素", "算力概念", "数据安全", "eSIM", "WiFi", "IPv6", "超清视频", "华为概念", "区块链", "5G概念", "量子通信", "阿里概念", "移动支付", "物联网"],
                      "600104.SH": ["大盘", "大市值", "换电概念", "固态电池", "半导体概念", "分拆预期", "万达概念", "超级品牌", "车联网", "新能源车", "无人驾驶", "一带一路", "国企改革", "燃料电池", "上海自贸"],
                      "600346.SH": ["大盘", "大市值", "降解塑料", "东北振兴", "智能机器", "锂电池"],
                      "600570.SH": ["大盘", "大市值", "多模态AI", "信创", "碳交易", "蚂蚁概念", "互联医疗", "生物识别", "区块链", "人工智能", "国产软件", "阿里概念", "电商概念", "互联金融", "参股保险", "云计算"],
                      "600887.SH": ["大盘", "大市值", "乳业", "新零售", "超级品牌", "婴童概念"], 
                      "601390.SH": ["大盘", "大市值", "央企改革", "工程机械", "磁悬浮", "装配建筑", "分拆预期", "万达概念", "债转股", "PPP模式", "一带一路", "小金属概念", "国企改革", "水利建设", "铁路基建"], 
                      "603160.SH": ["大盘", "大市值", "中芯概念", "AI手机", "星闪概念", "汽车芯片", "半导体概念", "传感器", "无线耳机", "华为概念", "小米概念", "生物识别", "国产芯片", "人工智能", "智能家居", "智能穿戴", "物联网", "深圳特区"],
                      "601668.SH": ["大盘", "大市值", "央企改革", "装配建筑", "数字孪生", "PPP模式", "一带一路"]}
    
        industry_news = []
        len_token = []
        for index, row in news_summary.iterrows():
            date_end = datetime.strptime(row['rec_time'], "%Y-%m-%d %H:%M:%S")
            data_start = date_end - timedelta(days=30)
            df_target_news = df_news[(df_news['datetime'] >= data_start) & (df_news['datetime'] <= date_end)].reset_index(drop=True)
            print(index, len(news_summary))
            accepted_news = []
            for word in stock_keywords[stock]:
                new_list, top_rank_score_and_idxs = retriever(model, word, df_target_news, score_weight=[0.7, 0.3], top_k=5)
                for index, item in enumerate(top_rank_score_and_idxs):
                    if item[1] > 0.4:
                        accepted_news.append(new_list[index])
            industry_news.append("\n".join(accepted_news))
            tokens = tokenizer.encode(industry_news[-1], truncation=True)
            print(industry_news[-1])
            len_token.append(len(tokens))
            
        news_summary.loc[:, "industry"] = industry_news
        news_summary.loc[:, 'industry_token_len'] = len_token
        
        table = pa.Table.from_pandas(news_summary)
        pq.write_table(table, '../data/announcement/' + stock + 'summary_with_industry.parquet')
#%%
