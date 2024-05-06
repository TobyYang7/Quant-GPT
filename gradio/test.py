import gradio as gr
import os
import pandas as pd
import time
from get_news import get_news
from llmtuner import ChatModel


def predict(message, history, temperature, top_p):
    final_model = ChatModel(dict(
        model_name_or_path=final_model_path,
        template='qwen',
        temperature=temperature,
        top_p=top_p,
        cutoff_len=8192,
        infer_backend='vllm',
        max_new_token=1024,
        vllm_gpu_util=0.7
    ))

    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": message})

    response = final_model.chat(history_openai_format, system='请对以下公告内容做投资情绪分析', stream=True)

    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message = partial_message + chunk.choices[0].delta.content
            yield partial_message
