# 导入所需的模块
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np


# import librosa
# import soundfile as sf


# 定义一个函数，用于读取图像文件，并将其转换为numpy数组
def read_image(filename):
    image = Image.open(filename)
    image = image.resize((300, 300))
    image = image.convert('RGB')
    image = np.array(image)
    return image


# 定义一个函数，用于在GUI上显示一个图像
def display_image(image, text, label):
    # 将图像数组转换为PIL图像对象
    image = Image.fromarray(image)
    # 将PIL图像对象转换为Tkinter图像对象
    image = ImageTk.PhotoImage(image)
    # 在标签上显示图像
    label.configure(text=text, image=image, compound=tk.TOP)
    label.image = image


# # 定义一个函数，用于在GUI上播放一个音频
# def play_audio(audio):
#     import subprocess
#     subprocess.call(r'C:\Program Files (x86)\Windows Media Player\wmplayer.exe /Users/amber/Desktop/0520/11emusic/output.wav')

def play_audio(audio):
    import os
    os.system('afplay /Users/amber/Desktop/0520/11emusic/output.wav')

# 定义一个函数，用于生成对应词列表的音频
def words2audio():
    import requests
    global words_list, audio
    if words_list is not None:
        def Generate(text):
            API_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-small"
            headers = {"Authorization": "Bearer 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"}

            def query(payload):
                response = requests.post(API_URL, headers=headers, json=payload)
                return response.content

            audio_bytes = query({
                "inputs": f"{text}",
            })
            from IPython.display import Audio
            return Audio(audio_bytes)

        audio = Generate(words_list)

        with open('output.wav', 'wb') as f:
            f.write(audio.data)

        messagebox.showinfo('提示', '转换成功，音频文件已保存为output.wav')


# 创建一个Tkinter窗口对象
window = tk.Tk()
# 设置窗口的标题
window.title('EMUSIC')
window.geometry('800x600')

# 创建一个标签，用于显示第一张图像
label1 = tk.Label(window)
label1.place(x=0, y=0, width=400, height=400)

# 创建一个标签，用于显示第二张图像
label2 = tk.Label(window)
label2.place(x=400, y=0, width=400, height=400)

# 创建一个标签，用于显示扩充后的词库
label3 = tk.Label(window, wraplength=500)
label3.place(x=150, y=450, width=550, height=50)

# 创建一个按钮，用于选择第一张图像
button1 = tk.Button(window, text='选择情绪图像', command=lambda: select_image(1))
button1.place(x=150, y=380, width=150, height=50)

# 创建一个按钮，用于选择第二张图像
button2 = tk.Button(window, text='选择场景图像', command=lambda: select_image(2))
button2.place(x=500, y=380, width=150, height=50)

# 创建一个按钮，用于生成音频
button3 = tk.Button(window, text='EMusic', command=words2audio)
button3.place(x=250, y=520, width=150, height=50)

# 创建一个按钮，用于播放音频
button4 = tk.Button(window, text='播放', command=lambda: play_audio(audio))
button4.place(x=400, y=520, width=150, height=50)

# 创建一个按钮，用于对输入词进行扩充
button5 = tk.Button(window, text='输入词扩充', command=lambda: get_words_list(label3))
button5.place(x=0, y=450, width=150, height=50)

# 定义一个变量，用于存储第一张图像的文件名
filename1 = None
# 定义一个变量，用于存储第二张图像的文件名
filename2 = None
# 定义一个变量，用于存储拼接后的图像数组
image = None
# 定义一个变量，用于存储音频信号
audio = None
emotion_word1 = None
emotion_word = None
scene_words = None
words_list = None


# 定义一个函数，用于选择图像
def select_image(n):
    # 使用全局变量
    global filename1, filename2, image, audio, emotion_word, scene_words
    # 弹出一个文件选择对话框，让用户选择一个图像文件
    filename = filedialog.askopenfilename(title='选择图像', filetypes=[('图像文件', '*.jpg *.png *.bmp')])
    # 如果用户选择了一个文件
    if filename:
        # 如果是选择第一张图像
        if n == 1:
            # 将文件名赋值给filename1
            filename1 = filename
            # 读取图像文件，并将其转换为numpy数组
            image1 = read_image(filename1)
            from facetest import emotion
            emotion_word, emotion_word1 = emotion(filename1)
            text1 = emotion_word
            # 在GUI上显示图像
            display_image(image1, "情绪词：" + text1, label1)
        # 如果是选择第二张图像
        elif n == 2:
            # 将文件名赋值给filename2
            filename2 = filename
            with open(filename2, "rb") as f:
                # 读取文件的所有内容
                data = f.read()
            # 创建一个新文件"input.jpg"，以二进制模式写入
            with open("input.jpg", "wb") as f:
                # 写入读取的内容
                f.write(data)
            import os
            os.system('python run_scene_attributeCNN.py')
            with open("scene_words.txt", "r") as file:
                content = file.read()
            scene_words = content.split(",")
            # 读取图像文件，并将其转换为numpy数组
            image2 = read_image(filename2)
            text2 = ', '.join(scene_words)
            # 在GUI上显示图像
            display_image(image2, "场景词：" + text2, label2)


def get_words_list(label):
    global emotion_word, scene_words, words_list
    from get_sim_90 import part3_main
    words_list = part3_main(emotion_word, scene_words)
    label.configure(text=words_list)
    words_list = scene_words + words_list
    return words_list


# 启动Tkinter的主循环
window.mainloop()
