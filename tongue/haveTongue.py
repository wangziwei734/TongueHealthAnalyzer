import tongue.segmentation_tongue as st
import cv2
import numpy as np

# 面积控制阈值
min_rate = 0.01
max_rate = 0.9

# 计算图像的亮度以及舌部所占整张图的比例
def calcuAera(imgPath):
    img = cv2.imread(imgPath)  # 读取图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 将图像转换为灰度图像
    sum = np.sum(gray)  # 对像素值求和
    average = sum / (gray.shape[0] * gray.shape[1]) # 求所有像素的平均值
    # 亮度控制
    if average < 50: # 如果像素的平均值小于50
        return -1, [] # 返回-1，表示图像过暗
    if average > 200: # 如果像素的平均值大于200
        return -2, [] # 返回-2，表示图像过亮
    # 获取舌部mask
    mask = st.seg_tongue(imgPath) # 调用定义的seg_tongue()函数，获取舌体mask
    x, y, w, h = cv2.boundingRect(mask) # 使用cv2.boundingRect()方法找到分割图像中的最小矩形框，并返回左上角坐标和宽高
    cut_gray = gray[y:y + h, x:x + w] # 从灰度图像中，提取矩形框中的区域
    # 计算舌部矩形内的颜色均值再进行一步亮度控制
    cut_sum = np.sum(cut_gray) #计算矩形框中像素的和
    if cut_sum !=0:
        cut_average = cut_sum / (cut_gray.shape[0] * cut_gray.shape[1]) # 计算矩形框中像素的平均值
        if cut_average < 50: # 如果像素的平均值小于50
            return -1, [] # 返回-1，表示图像过暗
        if cut_average > 200: # 如果像素的平均值大于200
            return -2, [] # 返回-2，表示图像过亮

    # 输出舌部区域矩形的范围
    min_x = x # 矩形框左上角的横坐标
    min_y = y # 矩形框左上角的纵坐标
    max_x = x + w + 1 # 矩形框右下角的横坐标
    max_y = y + h + 1 # 矩形框右下角的纵坐标
    if min_x < 0: # 如果矩形框左上角的横坐标小于0
        min_x = 0 # 则将其设置为0
    if min_y < 0: # 如果矩形框左上角的纵坐标小于0
        min_y = 0 # 则将其设置为0
    if max_x > img.shape[1]: # 如果矩形框右下角的横坐标大于图像的宽
        max_x = img.shape[1] # 则将其设置为宽的值
    if max_y > img.shape[0]: # 如果矩形框右下角的纵坐标大于图像的高
        max_y = img.shape[0] # 则将其设置为高的值
    return w*h/(mask.shape[0]*mask.shape[1]), [int(min_x), int(min_y), int(max_x), int(max_y)],mask # 返回舌体占整张图像的比例，舌体矩形的左上角和右下角的坐标，以及舌体mask


# 舌部质量检测
def haveTongue(imgPath):
    res, boxs, mask = calcuAera(imgPath)
    if res == -1:
        print('图像过暗，请重新拍摄')
        return 0, [], '图像过暗，请重新拍摄'
    if res == -2:
        print('图像过亮，请重新拍摄')
        return 0, [], '图像过亮，请重新拍摄'
    if res < min_rate or res > max_rate: #如果舌体占整张图像的比例小于设置的最小面积阈值，或者舌体占整张图像的比例大于设置的最大面积阈值，则表示未检测到舌体或者舌体不完整，该类图像质量检测不通过
        print('未检测到舌头，请重新拍摄')
        return 0, [],[], '未检测到舌头，请重新拍摄'
    return 1, boxs, mask, '图片质量通过'