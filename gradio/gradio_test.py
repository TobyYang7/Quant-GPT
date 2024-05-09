import gradio as gr
import os
import pandas as pd
import time
from get_news import get_news
from llmtuner import ChatModel
from RAG_base import retriever


def predict(query, history, temperature, top_p):
    final_model = ChatModel(dict(
        model_name_or_path=final_model_path,
        template='qwen',
        temperature=temperature,
        top_p=top_p,
        cutoff_len=8192,
        infer_backend='huggingface',
        max_new_tokens=1024,
        # vllm_gpu_util = 0.7
    ))

    history_openai_format = []
    # TODO: multi rounds conversation
    # for user, assistant in history:
    #     history_openai_format.append({"role": "user", "content": user})
    #     history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": query})

    response = final_model.chat(history_openai_format, system='你是一个金融交易员')
    return response


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


start_time = time.time()  # 获取开始时间

# 这里放置你想要测量运行时间的代码
time.sleep(1)  # 示例: 延时1秒

end_time = time.time()  # 获取结束时间
duration = end_time - start_time  # 计算持续时间

print(f"程序运行了 {duration} 秒")


def chatbot(query, history, time_period, temperature, top_p, check_box, file):
    # TODO: multi rounds conversation
    start_time = time.time()
    df_news = get_news(time_period)
    end_time = time.time()
    print(f"get_news运行了 {end_time - start_time} 秒")
    start_time = time.time()
    retrieval_news = retriever(query, embedding_model_path, df_news, score_weight=[0.7, 0.3], top_k=5)
    end_time = time.time()
    print(f"retriever运行了 {end_time - start_time} 秒")
    start_time = time.time()
    joined_news = '\n\n'.join(retrieval_news)
    combine_query = f'相关新闻内容如下：\n\n{joined_news}\n\n 根据以上新闻的内容对以下这则公告的投资情绪进行判断，你的回答只能为[极度悲观，相对悲观，中性，相对乐观，极度乐观]五个选项中的一个，公告如下：\n\n{query}\n\n 选项为：'
    print(combine_query)
    response = predict(combine_query, history, temperature, top_p)
    end_time = time.time()
    print(f"response运行了 {end_time - start_time} 秒")
    partial_message = f"相关新闻内容如下：\n\n{joined_news}\n\n\n投资情绪判断为："

    # for chunk in response:
    #     if chunk.choices[0].delta.content is not None:
    #           partial_message = partial_message + chunk.choices[0].delta.content
    #           yield partial_message
    return partial_message


if __name__ == "__main__":
    # embedding_model_path = '/home/JQpeng/workspace/embedding_model/BAAI-bge-m3'
    embedding_model_path = 'BAAI/bge-m3'
    final_model_path = '../exp_model/v2'
    # TODO:TongYi summary model
    summary_model_path = '/home/zhangmin/.cache/modelscope/hub/qwen/Qwen-7B-Chat'

    with gr.Blocks() as demo:
        with gr.Row():
            time_period = gr.Slider(minimum=1, maximum=400, value=5, step=1, label='Time Period', render=False, min_width=100)
            temperature = gr.Slider(minimum=0, maximum=1, value=0.8, step=0.05, label='Temperature', render=False)
            top_p = gr.Slider(minimum=0, maximum=1, value=0.8, step=0.05, label='Top_p', render=False)
            check_box = gr.CheckboxGroup(["公司", "国际", "A股", "宏观", "市场"], label="News Types", render=False, value=["公司", "国际", "A股", "宏观", "市场"])
            File = gr.File(label='Upload PDF', render=False)
        gr.ChatInterface(
            chatbot, chatbot=gr.Chatbot(height=460, render=False), additional_inputs=[time_period, temperature, top_p, check_box, File], additional_inputs_accordion="Custom Parameters",
        )

    demo.launch(server_name='0.0.0.0', share=False, inbrowser=True, server_port=16006)
