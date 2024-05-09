import os
import pdfplumber


def pdf2txt():
    # 源文件夹路径
    source_folder = "./test_doc"
    # 目标文件夹路径，用于保存TXT文件
    target_folder = "./output"

    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.endswith(".pdf"):
            # 构建完整的文件路径
            file_path = os.path.join(source_folder, filename)

            # 使用pdfplumber打开PDF文件
            with pdfplumber.open(file_path) as pdf:
                # 初始化一个空字符串来保存文本内容
                text = ""

                # 遍历PDF中的每一页
                for page in pdf.pages:
                    # 提取页面的文本并添加到text变量中
                    text += page.extract_text()
                    text += "\n\n"  # 添加换行符以分隔不同页面的内容

            # 构建目标TXT文件的路径，文件名保持不变，只是扩展名改为.txt
            txt_file_path = os.path.join(target_folder, filename.replace(".pdf", ".txt"))

            # 将文本内容写入TXT文件
            with open(txt_file_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(text)

            print(f"已转换文件: {filename} -> {txt_file_path}")