## 爬取B站音频，语音识别成文字，然后总结要点

import os
import sys
import requests
import re
import json
import subprocess
import argparse
import logging
from pydub import AudioSegment
import time
import shutil
import openai

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import SpacyTextSplitter
from llama_index import GPTListIndex, LLMPredictor, ServiceContext, SimpleDirectoryReader
from llama_index import Prompt
from llama_index.node_parser import SimpleNodeParser


openai.api_key = os.getenv("OPENAI_API_KEY")

# 缓存B站url音频
def download_audio(url, output_dir):
    download_dir = os.path.join(output_dir, "download")
    os.makedirs(download_dir, exist_ok=True)

    start_time = time.time() # 获取开始时间

    # 添加headers请求头，对Python解释器进行伪装
    # referer 和 User-Agent要改写成字典形式
    headers = {
        "referer":"https://www.bilibili.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    }

    # 用 requests 的 get 方法访问网页
    response = requests.get(url=url, headers=headers)

    # 返回响应状态码：<Response [200]>
    # print("返回200，则网页请求成功：",response)

    # .text获取网页源代码
    # print(response.text)

    # 提取视频标题
    # 调用 re 的 findall 方法，去response.text中匹配我们要的标题
    # 正则表达式提取的数据返回的是一个列表，用[0]从列表中取值
    video_title = re.findall('<h1 title="(.*?)"', response.text)[0]
    # 如果标题里有[\/:*?<>|]特殊字符，直接删除
    video_title = re.sub(r"[\/:*?<>|]","",video_title)
    print("视频标题：",video_title)

    html_data =  re.findall('<script>window.__playinfo__=(.*?)</script>', response.text)[0]

    # html_data是字符串类型，将字符串转换成字典
    json_data=json.loads(html_data)

    json_dicts = json.dumps(json_data,indent=4)

    # print(json_dicts)

    # # 提取视频画面网址
    # video_url = json_data["data"]["dash"]["video"][0]["baseUrl"]
    # print("视频画面地址为：", video_url)
    # 提取音频网址
    audio_url = json_data["data"]["dash"]["audio"][0]["baseUrl"]
    print("音频地址：", audio_url)

    print("\n音频缓存中.......")

    # response.content获取响应体的二进制数据
    audio_content = requests.get(url=audio_url,headers=headers).content

    # # 创建mp3文件，写入二进制数据到硬盘
    mp3_file = os.path.join(download_dir, video_title + ".mp3")
    with open (mp3_file, mode = "wb") as f :
        f.write(audio_content)

    print("音频缓存成功！")
    end_time = time.time() # 获取结束时间
    print("音频下载运行时长(秒): ", "{:.3f}".format(end_time - start_time))
    return mp3_file


def convert_to_mp3(input_file, output_dir):
    print("intput_file: ", input_file)
    print("\n格式转换中.......")
    start_time = time.time() # 获取开始时间
    valid_extensions = ['.mp3', '.mp4', '.m4a', '.wav', '.flac']
    
    # 检查文件扩展名是否在有效扩展名列表中
    file_extension = os.path.splitext(input_file)[1].lower()
    if file_extension not in valid_extensions:
        print(f"{input_file} 不是有效的音视频文件，跳过转换。")
        return

    mp3_dir = os.path.join(output_dir, "mp3")
    os.makedirs(mp3_dir, exist_ok=True)

    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1) # 单通道
    mp3_file = os.path.join(mp3_dir, os.path.splitext(os.path.basename(input_file))[0] + ".mp3")
    audio.export(mp3_file, format="mp3")
    print(f"{input_file} 已转换为 MP3 格式，并保存到 {mp3_file}")
    end_time = time.time() # 获取结束时间
    print("格式转换运行时长(秒): ", "{:.3f}".format(end_time - start_time))
    return mp3_file


def split_and_transcribe_mp3(input_file, output_dir):
    print("\n音频切分中.......")
    start_time = time.time() # 获取开始时间
    audio = AudioSegment.from_mp3(input_file)
    duration_ms = len(audio)

    split_dir = os.path.join(output_dir, "split")
    os.makedirs(split_dir, exist_ok=True)

    chunk_length_ms = 900000  # 15分钟的毫秒数
    num_chunks = (duration_ms + chunk_length_ms - 1) // chunk_length_ms
    print("num_chunks: ", num_chunks)

    if num_chunks == 0:
        # 如果文件长度不足15分钟，直接复制原文件到目标目录
        output_file = os.path.join(split_dir, os.path.basename(input_file))
        audio.export(output_file, format="mp3")
        print(f"{input_file} 复制到 {output_file}")
        transcribe_audio(output_file, output_dir)
    else:
        for i in range(num_chunks):
            chunk_start_time = i * chunk_length_ms
            chunk_end_time = min((i + 1) * chunk_length_ms, duration_ms)
            chunk = audio[chunk_start_time:chunk_end_time]
            chunk_file = os.path.join(split_dir, os.path.splitext(os.path.basename(input_file))[0] + f"_part_{i+1}.mp3")
            chunk.export(chunk_file, format="mp3")
            print(f"音频片断 {i+1} 已保存到 {chunk_file}")
            transcribe_audio(chunk_file, output_dir)
    end_time = time.time() # 获取结束时间
    print("音频切分运行时长(秒): ", "{:.3f}".format(end_time - start_time))
    return duration_ms

def process_directory(input_dir, output_dir):
    mp3_dir = os.path.join(output_dir, "mp3")
    os.makedirs(mp3_dir, exist_ok=True)

    total_duration_ms = 0  # 用于累计所有 MP3 文件的总时长

    for root, _, files in os.walk(input_dir):
        print("files: ", files)
        for file in files:
            file_path = os.path.join(root, file)
            
            # 转换为 MP3文件，单通道
            mp3_output_file = convert_to_mp3(file_path, output_dir)

            # 切分 MP3 文件
            total_duration_ms += split_and_transcribe_mp3(mp3_output_file, output_dir)           
    return total_duration_ms

def transcribe_audio(input_file, output_dir):
    print("\n语音识别中.......")
    start_time = time.time() # 获取开始时间
    try:
        srt_dir = os.path.join(output_dir, "srt")
        os.makedirs(srt_dir, exist_ok=True)
        txt_dir = os.path.join(output_dir, "txt")
        os.makedirs(txt_dir, exist_ok=True)
        # 打开音频文件
        with open(input_file, "rb") as audio_file:
            # 调用 OpenAI API 进行语音识别
            prompt = ""
            transcript = openai.Audio.transcribe("whisper-1", audio_file, response_format="srt", prompt=prompt)

        # 获取输入文件的基本文件名
        base_file_name = os.path.basename(input_file)

        # 将识别结果写入 SRT 文件，包含时间信息
        srt_file = os.path.splitext(base_file_name)[0] + ".srt"
        srt_file_path = os.path.join(srt_dir, srt_file)
        with open(srt_file_path, "w", encoding='utf-8') as f:
            f.write(transcript)

        # 保存字幕文本到 TXT 文件，不包含时间信息
        txt_file = os.path.splitext(base_file_name)[0] + ".txt"
        txt_file_path = os.path.join(txt_dir, txt_file)
        with open(txt_file_path, "w", encoding='utf-8') as f:
            lines = transcript.strip().split("\n")
            content = "\n".join(lines[2::4])  # 跳过时间行，仅保存文本行
            f.write(content)
                
        print(f"语音识别结果已保存到 {srt_file_path}")
        end_time = time.time() # 获取结束时间
        print("语音识别运行时长(秒): ", "{:.3f}".format(end_time - start_time))
    
    except Exception as e:
        print(f"语音识别时发生错误: {e}")


# 用ChatGPT总结
def summary_srt_files(output_dir):

    srt_dir = os.path.join(output_dir, "srt")
    if not os.path.exists(srt_dir):
        print(f"错误: {srt_dir} 目录不存在。请先执行转换和语音识别操作。")
        return
    
    print("\n文本总结中.......")

    start_time = time.time() # 获取开始时间

    # define LLM
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=1024))

    text_splitter = SpacyTextSplitter(pipeline="zh_core_web_sm", chunk_size = 4096)
    parser = SimpleNodeParser(text_splitter=text_splitter)
    documents = SimpleDirectoryReader(srt_dir).load_data()
    nodes = parser.get_nodes_from_documents(documents)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    list_index = GPTListIndex(nodes=nodes, service_context=service_context)
    
    # tree_summarize （总结摘要-最优）
    # simple_summarize （最省token）
    # refine （基于关键词询问-最优）
    # compact（较省token）
    query_engine = list_index.as_query_engine(response_mode="tree_summarize")

    # response = query_engine.query("请你总结一下内容，要点限定在5条以内，并以Markdown格式显示")
    prompt = "请你总结一下内容要点，尽量简洁！"
    response = query_engine.query(prompt)

    # 总结
    print("视频要点：", response)
    end_time = time.time() # 获取结束时间
    print("文本总结运行时长(秒): ", "{:.3f}".format(end_time - start_time))

    return response


# url
# "https://www.bilibili.com/video/BV1Z4411179m/?spm_id_from=333.999.0.0"
# "https://www.bilibili.com/video/BV1xE411B7V4/?spm_id_from=333.337.search-card.all.click"
def main():
    if len(sys.argv) != 2:
        print("用法: python audio_gpt.py [URL|音频文件|音频目录]")
        return

    input_arg = sys.argv[1]
    # input_arg = f"D:\\project\\prjGPT\\src\\udio_gpt\\audio_preview\\raw1\\download"
    input_dir = ""

    current_time = time.strftime("%Y%m%d%H%M%S")
    
    # 使用当前时间作为输出目录名
    output_dir = os.path.abspath(os.path.join(os.getcwd(), current_time))
    os.makedirs(output_dir, exist_ok=True)

    total_duration_ms = 0  # 用于累计所有 MP3 文件的总时长

    start_time = time.time() # 获取开始时间
    if os.path.isdir(output_dir):
        if input_arg.startswith("http"): # 如果是url，则先下载
            audio_file = download_audio(input_arg, output_dir)
            input_dir = audio_file
        else:
            input_dir = input_arg

        print("input_dir: ", input_dir)
        if os.path.isfile(input_dir):
            mp3_file = convert_to_mp3(input_dir, output_dir)
            if mp3_file:
                total_duration_ms = split_and_transcribe_mp3(mp3_file, output_dir)
        elif os.path.isdir(input_dir):
            total_duration_ms = process_directory(input_dir, output_dir)
    else:
        print("输出目录不存在或不是一个有效的目录。")

    summary_srt_files(output_dir)
    end_time = time.time() # 获取结束时间

    # 计算总时长
    print("音频总时长(秒): ", (total_duration_ms/1000))
    print("运行总时长(秒): ", "{:.3f}".format(end_time - start_time))

if __name__ == "__main__":
    main()
