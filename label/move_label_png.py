# 导入包
import cv2
import numpy as np
import os
import shutil

path1 = 'label_json/'
count = 1
for file_1 in sorted(os.listdir(path1)):  # 遍历label_json/目录下的所有子目录
    path2 = os.path.join(path1, file_1)  # 组合路径，返回子目录的路径
    if os.path.isdir(path2):
        print(count)
        count += 1
        print(path2)
        path_save = os.path.join('label_png', file_1[:-5] + '.png')
        img_path = path2 + '/label.png'
        shutil.copyfile(img_path, path_save)  # 将图像标签复制到label_png目录下
