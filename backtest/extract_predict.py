#%%
import pandas as pd
import json

def extract_json_predict_result(data):
    extracted_data = []
    
    stock_mapping = {
        "三一重工": "600031.SH",
        "招商银行": "600036.SH",
        "中国联通": "600050.SH",
        "上汽集团": "600104.SH",
        "恒力石化": "600346.SH",
        "恒生电子": "600570.SH",
        "伊利股份": "600887.SH",
        "中国中铁": "601390.SH",
        "汇项科技": "603160.SH",
        "中国建筑": "601668.SH"
    }
    
    for item in data:
        time = item["content"].split("现在是")[1].split("\n")[0]
        predicted = item["predicted"]
        stock_code = stock_mapping[item["prompt"].split("预测")[1].split("股票")[0]]

        extracted_data.append({"predicted": predicted, "time": time, "stock_code": stock_code})
    
    df = pd.DataFrame(extracted_data)

    return df

#%%
if __name__ == '__main__':
    file_path = "/home/zhangmin/toby/Quant-GPT/data/test_dataset/v4_cp1800.json"

    with open(file_path, "r") as file:
        data = json.load(file)
    
    predict_result = extract_json_predict_result(data)
# %%
    predict_result.sort_values(by = "time", inplace = True)
    predict_result.reset_index(inplace = True, drop = True)
    predict_result.to_csv("./result/Quant-GPT.csv")
# %%
