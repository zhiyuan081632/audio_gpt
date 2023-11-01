import tkinter as tk
from tkinter import filedialog
import subprocess
import threading
import sys
import os


class AudioGPTWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("B站音频处理工具")

        self.input_label = tk.Label(root, text="请输入URL或选择音频文件目录：")
        self.input_label.pack()

        self.input_entry = tk.Entry(root)
        self.input_entry.pack()

        self.browse_button = tk.Button(root, text="选择目录", command=self.browse_directory)
        self.browse_button.pack()

        self.run_button = tk.Button(root, text="开始处理", command=self.start_processing)
        self.run_button.pack()

        self.status_label = tk.Label(root, text="")
        self.status_label.pack()

    def browse_directory(self):
        directory = filedialog.askdirectory()
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, directory)

    def start_processing(self):
        input_arg = self.input_entry.get()
        if not input_arg:
            self.status_label.config(text="请输入URL或选择音频文件目录。")
            return

        self.status_label.config(text="处理中，请稍候...")
        self.run_button.config(state=tk.DISABLED)

        # 创建一个新线程来运行处理任务，以避免阻塞GUI
        processing_thread = threading.Thread(target=self.run_audio_gpt, args=(input_arg,))
        processing_thread.start()

    def run_audio_gpt(self, input_arg):
        # 调用audio_gpt.py来执行任务
        try:
            subprocess.run(["python", "audio_gpt.py", input_arg])
            self.status_label.config(text="处理完成！")
        except Exception as e:
            self.status_label.config(text=f"处理时发生错误: {str(e)}")
        finally:
            self.run_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioGPTWindow(root)
    root.mainloop()
