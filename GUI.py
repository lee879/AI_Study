import os
import numpy as np
from tkinter import *
import tkinter
from detect_facemask import run
#from keras.models import load_model
#from keras.preprocessing.image import img_to_array
from tkinter.filedialog import askopenfilename,askdirectory
import cv2
import matplotlib.pyplot as plt
#from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image, ImageTk
import time
class GUI:
    def __init__(self):
        self.image_path = ""
        self.file_path_1 = ''
        self.file_path = ''
        self.emotion_list = []
        self.win = tkinter.Tk()  # 构造窗体
        self.win.title("人脸口罩佩戴检测软件")  # 标题
        self.win.geometry("800x500+500+500")  # 设置窗口的大小,具体方法在上面的注释已说明
        self.main()
        self.win.mainloop()
    def re_win(self):
        self.win = tkinter.Tk()  # 构造窗体
        self.win.title("人脸口罩佩戴检测软件")  # 标题
        self.win.geometry("800x500+500+500")  # 设置窗口的大小,具体方法在上面的注释已说明
        self.main()
    def main(self):
        self.txt_1 = tkinter.Label(self.win, text='功能模块:', font=('粗体', 20))
        self.txt_1.pack()
        self.txt_1.place(x=100, y=50)
        self.button_1 = tkinter.Button(self.win, text="图片识别", width=14, height=2, font=('粗体', 12),
                                       command=lambda:[self.win.destroy(),self.createSecondwin()])
        self.button_1.pack()
        self.button_1.place(x=250, y=100)
        self.button_2 = tkinter.Button(self.win, text="视频识别", width=14, height=2, font=('粗体', 12),
                                       command=lambda:[self.win.destroy(),self.createThirdwin()])
        self.button_2.pack()
        self.button_2.place(x=250, y=150)
        self.button_3 = tkinter.Button(self.win, text="摄像头识别", width=14, height=2, font=('粗体', 12),
                                       command=lambda: [self.win.destroy(), self.createfourwin()])
        self.button_3.pack()
        self.button_3.place(x=250, y=200)
    def createSecondwin(self):
        self.win2 = tkinter.Tk()  # 构造窗体
        self.win2.title("图片识别")  # 标题
        self.win2.geometry("800x500+500+500")  # 设置窗口的大小,具体方法在上面的注释已说明
        txt_3 = tkinter.Label(self.win2, text='图片所在路径', font=('粗体', 20))
        txt_3.pack()
        txt_3.place(x=50, y=40)
        button_4 = tkinter.Button(self.win2, text="返回主界面", width=14, height=2, font=('粗体', 12),
                                  command=lambda: [self.win2.destroy(),self.re_win(), self.main()])
        button_4.pack()
        button_4.place(x=650, y=450)
        button_5 = tkinter.Button(self.win2, text="浏览", width=10, height=2, font=('粗体', 12),command=lambda:self.openimage())
        button_5.pack()
        button_5.place(x=500, y=80)
        button_6 = tkinter.Button(self.win2, text="执行", width=10, height=2, font=('粗体', 12),command=lambda: self.pre_img())
        button_6.pack()
        button_6.place(x=600, y=80)
        txt_5 = tkinter.Label(self.win2, text="原始图片", font=('粗体', 12))
        txt_5.pack()
        txt_5.place(x=150, y=400)
        txt_6 = tkinter.Label(self.win2, text="输出图片", font=('粗体', 12))
        txt_6.pack()
        txt_6.place(x=530, y=400)
    def openimage(self):
        self.image_path = askopenfilename()
        txt_4 = tkinter.Label(self.win2, text=str(self.image_path), font=('粗体', 12), relief=tkinter.SUNKEN)
        txt_4.pack()
        txt_4.place(x=80, y=100)
    def openfile(self):
        self.save_video_path = askdirectory()
        txt_4 = tkinter.Label(self.win2, text=str(self.save_video_path), font=('粗体', 12), relief=tkinter.SUNKEN)
        txt_4.pack()
        txt_4.place(x=80, y=280)
    def pre_img(self):
        image_2 = Image.open(self.image_path)
        image_2 = image_2.resize((200, 200), Image.ANTIALIAS)
        photo_2 = ImageTk.PhotoImage(image=image_2)
        image_2 = tkinter.Label(self.win2, image=photo_2)
        image_2.image = photo_2
        image_2.pack()
        image_2.place(x=100, y=180)
        result_img = run(source=self.image_path, new_save_path="./test_out/")
        r_image = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        r_image = r_image.resize((200, 200), Image.ANTIALIAS)
        photo_3 = ImageTk.PhotoImage(image=r_image)
        r_image = tkinter.Label(self.win2, image=photo_3)
        r_image.image = photo_3
        r_image.pack()
        r_image.place(x=460, y=180)
    def createfourwin(self):
        self.win3 = tkinter.Tk()  # 构造窗体
        self.win3.title("摄像头识别")  # 标题
        self.win3.geometry("800x500+500+500")  # 设置窗口的大小,具体方法在上面的注释已说明
        self.camera = cv2.VideoCapture(0)
        self.canvas = Canvas(self.win3, width=600, height=400)
        self.canvas.pack()
        self.Flag_show = True
        button_4 = tkinter.Button(self.win3, text="返回主界面", width=14, height=2, font=('粗体', 12),
                                  command=lambda: [self.backFirst(),self.win3.destroy(), self.re_win(), self.main()])
        button_4.pack()
        button_4.place(x=650, y=450)
        while self.Flag_show:
            self.imgtk = self.emotion_video()
            self.canvas.create_image(0, 0, anchor='nw', image=self.imgtk)
            self.win3.update()
            self.win3.after(1)
    def emotion_video(self):
        result_img = run(source=0)
        print(result_img)
        r_image = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        r_image = r_image.resize((400, 400), Image.ANTIALIAS)
        self.imgtk = ImageTk.PhotoImage(image=r_image)
        return self.imgtk
    def backFirst(self):
        # 释放摄像头资源
        self.Flag_show =False
        self.camera.release()
        cv2.destroyAllWindows()
    def createThirdwin(self):
        self.win2 = tkinter.Tk()  # 构造窗体
        self.win2.title("视频识别")  # 标题
        self.win2.geometry("800x500+500+500")  # 设置窗口的大小,具体方法在上面的注释已说明
        txt_3 = tkinter.Label(self.win2, text='视频所在路径', font=('粗体', 20))
        txt_3.pack()
        txt_3.place(x=50, y=40)
        txt_5 = tkinter.Label(self.win2, text="视频保存路径", font=('粗体', 20))
        txt_5.pack()
        txt_5.place(x=50, y=200)
        button_4 = tkinter.Button(self.win2, text="返回主界面", width=14, height=2, font=('粗体', 12),
                                  command=lambda: [self.win2.destroy(), self.re_win(), self.main()])
        button_4.pack()
        button_4.place(x=650, y=450)
        button_5 = tkinter.Button(self.win2, text="选择视频", width=10, height=2, font=('粗体', 12),
                                  command=lambda: self.openimage())
        button_5.pack()
        button_5.place(x=500, y=80)
        button_7 = tkinter.Button(self.win2, text="选择保存路径", width=15, height=2, font=('粗体', 12),
                                  command=lambda: self.openfile())
        button_7.pack()
        button_7.place(x=500, y=260)
        button_6 = tkinter.Button(self.win2, text="执行", width=10, height=2, font=('粗体', 12),
                                  command=lambda: self.pre_video())
        button_6.pack()
        button_6.place(x=600, y=80)
    def pre_video(self):
        result_img = run(source=self.image_path , new_save_path=self.save_video_path)
        txt_8 = tkinter.Label(self.win2, text="已完成", font=('粗体', 12))
        txt_8.pack()
        txt_8.place(x=620, y=130)
    def draw_flag_1(self):
        self.draw_flag = True
        if self.draw_flag:
            now = time.localtime()
            now_time = time.strftime("%Y-%m-%d %H:%M:%S", now)
            now_time = now_time.split(" ",1)[0]+"_"+now_time.split(" ",1)[1].split(":",1)[0]+\
                       now_time.split(" ",1)[1].split(":",2)[1]+"_"+now_time.split(" ",1)[1].split(":",2)[2]
            emotion_num = len(self.emotion_list)
            emotion_num_list = list(range(1, emotion_num + 1))
            plt.plot(emotion_num_list, self.emotion_list, lw=1)
            plt.savefig("./emotion_record/"+str(now_time)+"_emotion.png")
    def del_emotion_list(self):
        self.emotion_list = []
GUI()

