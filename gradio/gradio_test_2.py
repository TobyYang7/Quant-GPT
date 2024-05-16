import gradio as gr
import os
import pandas as pd
import time
from get_news import get_news
from llmtuner import ChatModel
from RAG_base import retriever
import datetime
from datetime import datetime, timedelta
# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from FlagEmbedding import BGEM3FlagModel
import pdfplumber
from dateutil import parser
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
import Levenshtein


def query_recognition(query, history):
    print('Start query_recognition')

    prompt = f'请根据用户输入的内容判断其类别。如果输入内容是一则公司公告，请回复公司公告；如果内容不是公司公告，请回复日常问答。您的回答只能严格为[公司公告,日常问答]中的一个\
以下是一些示例，以帮助您区分:\
公司公告示例: 浙江华铁应急设备科技股份有限公司发布了一份关于对外担保进展的公告。主要内容包括：公司新增对外担保金额合计111,791.16万元，其中担保人包括华铁大黄蜂、华铁应急、华铁宇硕、浙江大黄蜂等，被担保人包括华铁供应链、浙江吉通、华铁应急等。\
日常问答示例: 你好，今天的天气怎么样？\
现在请根据以下用户输入内容进行判断：\
用户的输入为：{query} \
您的回复为：'

    response = predict_chatmodel(prompt, history)
    print(response)
    similarity_to_phrase1 = Levenshtein.distance(response, '日常问答')
    similarity_to_phrase2 = Levenshtein.distance(response, '公司公告')
    if similarity_to_phrase1 <= similarity_to_phrase2:
        return '日常问答'
    else:
        return '公司公告'


def pdf2str(file_path):
    print('Start pdf2str')
    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n\n"
    return text


def calculate_token_length(text):
    global tokenizer
    tokens = tokenizer.encode(text, truncation=True)
    return len(tokens)


def get_summary(query):
    print('Start get_summary')
    original_token_count = calculate_token_length(query)
    if original_token_count >= 70 * 1024:
        mid_point = len(query) // 2
        while query[mid_point] != ' ' and mid_point < len(query):
            mid_point += 1
        part1 = get_summary(summary_model, query[:mid_point])
        part2 = get_summary(summary_model, query[mid_point:])
        res = part1 + ' ' + part2
        print(res)
        return res
    else:
        query = "请您帮忙总结下面的内容，尽可能不损失对投资有用的信息。\n" + query
        res = predict_glm(query, [])   # history = []
        print(res)
        return res


# TODO: 新闻类型筛选
def get_news(time_period, date_end=datetime.now()):
    print('Start get_news')
    df_news['datetime'] = pd.to_datetime(df_news['datetime'])
    data_start = date_end - timedelta(days=time_period)
    df_target_news = df_news[(df_news['datetime'] >= data_start) & (df_news['datetime'] <= date_end)].reset_index(drop=True)
    return df_target_news


def retriever(query, df_news, score_weight=[0.7, 0.3], top_k=5):
    print('Start retriever')
    dense_vecs = df_news['dense_vecs']
    lexical_weights = df_news['lexical_weights']               # ！！！！！！！！！！测试feather数据结构
    query_embedding = embedding_model.encode(query, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    score_list = []
    for i in range(len(dense_vecs)):
        score_dense = query_embedding['dense_vecs'] @ dense_vecs[i].T
        filtered_dict = {k: v for k, v in lexical_weights[i].items() if v is not None}
        score_lexical = embedding_model.compute_lexical_matching_score(query_embedding['lexical_weights'], filtered_dict)
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


def predict_Qwen_new(query, history, temperature, top_p):
    print('Start predict_Qwen_new')
    final_model.generation_config = GenerationConfig.from_pretrained(final_model_path, temperature=temperature, top_p=top_p)

    # history_openai_format = []
    # TODO: multi rounds conversation
    # for user, assistant in history:
    #     history_openai_format.append({"role": "user", "content": user})
    #     history_openai_format.append({"role": "assistant", "content":assistant})
    # history_openai_format.append({"role": "user", "content": query})
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": query})

    # messages = [
    #     {"role": "system", "content": "你是一个金融交易员。"},
    #     {"role": "user", "content": query}
    # ]
    text = final_model_tokenizer.apply_chat_template(
        history_openai_format,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = final_model_tokenizer([text], return_tensors="pt").to('cuda')

    generated_ids = final_model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = final_model_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def predict_chatmodel(query, history, temperature=0.8, top_p=0.8):
    print('Start predict_chatmodel')
    final_model.generation_config = GenerationConfig.from_pretrained(final_model_path, temperature=temperature, top_p=top_p)
    history_format = []
    for human, assistant in history:
        history_format.append({"role": "user", "content": human})
        history_format.append({"role": "assistant", "content": assistant})
    history_format.append({"role": "user", "content": query})
    # messages = [{"role": "user", "content": query}]
    try:
        with torch.no_grad():
            resp = final_model.chat(history_format, system="你是一个金融交易员")
            response_text = resp[0].response_text
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print('WARNING: out of memory')
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            raise exception
        else:
            raise
    return response_text


def predict_glm(query, history):
    print('Start predict_glm')
    history_format = []
    for human, assistant in history:
        history_format.append({"role": "user", "content": human})
        history_format.append({"role": "assistant", "content": assistant})
    history_format.append({"role": "user", "content": query})
    # messages = [{"role": "user", "content": query}]
    try:
        with torch.no_grad():
            resp = summary_model.chat(history_format, system="你是一个金融交易员")
            response_text = resp[0].response_text
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print('WARNING: out of memory')
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            raise exception
        else:
            raise
    return response_text


def predict_openai(query, history, temperature, top_p):
    client = AzureOpenAI(
    )

    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model="XMGI-Chat4-GPT4",
        messages=history_openai_format,
        temperature=temperature,
        top_p=top_p,
        stream=True)

    return response


def chatbot(query, history, ann_time, time_period, temperature, top_p, check_box, file):
    if file is not None:
        file_content = pdf2str(file)
        query = file_content + '\n\n' + query
    is_ann = query_recognition(query, history)
    print(is_ann)
    if is_ann == '公司公告':
        # 多格式自动读取为datetime
        ann_time = parser.parse(ann_time)
        # ann_time = datetime.strptime(ann_time, "%Y-%m-%d %H:%M:%S")
        df_news = get_news(time_period, ann_time)

        summary_query = get_summary(query)
        retrieval_news = retriever(summary_query, df_news, score_weight=[0.7, 0.3], top_k=5)

        joined_news = '\n\n'.join(retrieval_news)
        combine_query = f'相关新闻内容如下：\n\n{joined_news}\n\n 根据以上新闻的内容对以下这则公告的投资情绪进行判断，你的回答只能为[极度悲观，相对悲观，中性，相对乐观，极度乐观]五个选项中的一个，公告如下：\n\n{summary_query}\n\n 选项为：'

        # response = predict_Qwen_new(combine_query, history, temperature, top_p)
        response = predict_chatmodel(combine_query, history, temperature, top_p)
        partial_message = f"相关新闻内容如下：\n\n{joined_news}\n\n\n投资情绪判断为：{response}"
    else:
        partial_message = predict_glm(query, history)
    # TODO:Streamer
    # for chunk in response:
    #     if chunk.choices[0].delta.content is not None:
    #           partial_message = partial_message + chunk.choices[0].delta.content
    #           yield partial_message
    return partial_message


if __name__ == "__main__":
    # embedding_model_path = '/home/JQpeng/workspace/embedding_model/BAAI-bge-m3'
    embedding_model_path = 'BAAI/bge-m3'
    final_model_path = '../exp_model/v2'
    # final_model_path = '/home/YJhou/workspace/llm_weight/Test/Qwen1.5-14B-Chat'
    # summary_model_path = '/home/zhangmin/.cache/modelscope/hub/qwen/Qwen-7B-Chat'
    summary_model_path = '/home/zhangmin/.cache/modelscope/hub/ZhipuAI/chatglm3-6b-128k'
    file_path = 'A_news_train_last1000_withvector.feather'

    # final_model_tokenizer = AutoTokenizer.from_pretrained(final_model_path, trust_remote_code=True)
    # final_model = AutoModelForCausalLM.from_pretrained(
    #     final_model_path,
    #     device_map="auto",
    #     torch_dtype="auto"
    #     )
    tokenizer = AutoTokenizer.from_pretrained(summary_model_path, trust_remote_code=True)
    final_model = ChatModel(dict(
        model_name_or_path=final_model_path,
        template="qwen",
        temperature=0.8,
        top_p=0.8,
        cutoff_len=8192,
        do_sample=True,
        max_new_tokens=1024,
        adapter_name_or_path="/home/zhangmin/toby/Quant-GPT/train/saves/v6_cp300",
        # infer_backend="vllm",
        # vllm_gpu_util=0.7
    ))
    print('final_model loaded successfully')

    summary_model = ChatModel(dict(
        model_name_or_path=summary_model_path,
        template="chatglm3",
        temperature=0.6,
        top_p=0.8,
        cutoff_len=8192,
        do_sample=True,
        max_new_tokens=1024
        # infer_backend="vllm",
        # vllm_gpu_util=0.7
    ))
    print('summary_model loaded successfully')

    embedding_model = BGEM3FlagModel(embedding_model_path, use_fp16=True)
    print('embedding_model loaded successfully')

    # df_news_train = pd.read_feather('/home/zhangmin/toby/Quant-GPT/gradio/A_news_train_all_withvector.feather')
    # df_news_test = pd.read_feather('/home/zhangmin/toby/Quant-GPT/gradio/A_news_test_all_withvector.feather')
    # df_news = pd.concat([df_news_train, df_news_test]).reset_index(drop=True)
    df_news = pd.read_feather(file_path)
    print('df_news loaded successfully')

    with gr.Blocks() as demo:
        with gr.Row():
            ann_time = gr.Textbox(value=f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', label='Announcement Time', render=False)
            time_period = gr.Slider(minimum=1, maximum=90, value=5, step=1, label='Time Period', render=False, min_width=100)
            temperature = gr.Slider(minimum=0, maximum=1, value=0.8, step=0.05, label='Temperature', render=False)
            top_p = gr.Slider(minimum=0, maximum=1, value=0.8, step=0.05, label='Top_p', render=False)
            check_box = gr.CheckboxGroup(["公司", "国际", "A股", "宏观", "市场"], label="News Types", render=False, value=["公司", "国际", "A股", "宏观", "市场"])
            File = gr.File(label='Upload PDF', file_count='single', render=False)
        gr.ChatInterface(
            chatbot, chatbot=gr.Chatbot(height=460, render=False), additional_inputs=[ann_time, time_period, temperature, top_p, check_box, File], additional_inputs_accordion="Custom Parameters",
        )

    # demo.launch(server_name='0.0.0.0', share=False, inbrowser=True, server_port=16006)
    demo.launch(server_name='0.0.0.0', share=True, inbrowser=True, server_port=8088)
    # demo.launch(share=True)
