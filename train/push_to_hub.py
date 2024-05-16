from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = 'ef53e3ed-cd68-4c13-aa47-1c12bf7f1dbc'

api = HubApi()
api.login(YOUR_ACCESS_TOKEN)
api.push_model(
    model_id="TobyYang7/Quant-GPT",
    model_dir="../exp_model/v3_2"
)
# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "BAAI/bge-m3"
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
