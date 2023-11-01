# 软件介绍
爬取B站音频，语音识别成文字，然后总结要点
* 可爬取B站、Youtube等视频
* 可以调用在线Whisper识别，也可以调用本地Whisper
* 在识别本地音频或音频目录

# 操作方法
* 首先拷贝一份config.example.json，修改为config.json
* 在config.json中填入你的OPENAI_API_KEY
* 运行命令：
  ```
  python audio_gpt.py [url|file|direction]
  ```

# 注意事项
* 如果输入url后无法下载，可以重新启动软件再试一下（url解析问题待修复）
