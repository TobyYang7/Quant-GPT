import gradio as gr
import os
import pandas as pd
import time
from get_news import get_news
# from llmtuner import ChatModel
from openai import AzureOpenAI
from RAG_base import retriever


def predict(query, history, temperature, top_p):
    final_model = ChatModel(dict(
        model_name_or_path = final_model_path,
        template = 'qwen',
        temperature = temperature,
        top_p = top_p,
        cutoff_len = 8192,
        infer_backend = 'vllm',
        max_new_token = 1024,
        vllm_gpu_util = 0.7
    ))
    
    history_openai_format = []
    # TODO: multi rounds conversation
    # for user, assistant in history:
    #     history_openai_format.append({"role": "user", "content": user})
    #     history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": query})
  
    response = final_model.chat(history_openai_format, system = '请对以下公告内容做投资情绪分析', stream=True)
    return response
    




def predict_openai(query, history, temperature, top_p):
    client = AzureOpenAI()
    
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": query})
  
    response = client.chat.completions.create(
        model="XMGI-Chat4-GPT4",
        messages= history_openai_format,
        temperature=temperature,
        top_p = top_p,
        stream=True)
    
    return response



def chatbot(query, history, time_period, temperature, top_p, check_box, file):
    # TODO: multi rounds conversation
    df_news = get_news(time_period)
    retrieval_news = retriever(query, embedding_model_path, df_news, score_weight=[0.7, 0.3], top_k=5)
    joined_news = '\n\n'.join(retrieval_news)
    combine_query = f'相关新闻内容如下：\n\n{joined_news}\n\n 根据以上新闻的内容对以下这则公告的投资情绪进行分析，公告如下：\n\n{query}'
    response = predict(combine_query, history, temperature, top_p)
    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
              partial_message = partial_message + chunk.choices[0].delta.content
              yield partial_message


if __name__ == "__main__":
    embedding_model_path = ''
    final_model_path = '/home/JQpeng/compspace/LLM_model/Qwen1.5-14B-Chat'
    # TODO:TongYi summary model
    summary_model_path = ''
    
    with gr.Blocks() as demo:
        with gr.Row():
            time_period = gr.Slider(minimum=1, maximum=30, value=5, step=1, label='time period',render=False)
            temperature = gr.Slider(minimum=0, maximum=1, value=0.8, step=0.05, label='temperature',render=False)
            top_p = gr.Slider(minimum=0, maximum=1, value=0.8, step=0.05, label='top_p',render=False)
            check_box = gr.CheckboxGroup(["公司", "国际", "A股", "宏观", "市场"], label="新闻类型", render=False, value=["公司", "国际", "A股", "宏观", "市场"])
            File = gr.File(label='Upload PDF', render=False)
        gr.ChatInterface(
                chatbot,chatbot=gr.Chatbot(height=300,render=False),additional_inputs=[time_period, temperature, top_p, check_box, File], additional_inputs_accordion="Custom Parameters", title="Quant GPT",
        )

    demo.launch()