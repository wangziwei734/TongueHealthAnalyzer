import numpy as np
import cv2
from scipy import signal
from tongue import segmentation_tongue


'''根据直方图得到最大值（峰值）、标准差宽度、均值、截断均值'''
def getStatistics(hist):  #hist：各通道的直方图
    max_a = signal.find_peaks(hist, distance=250)[0] #使用signal.find_peaks()方法找到直方图的峰值
    # 计算标准差
    std_a = hist.std() # 计算标准差
    idx_a = np.abs(hist - std_a).argmin()  # 计算最接近标准差的值
    stdWidth_a = np.abs(max_a - idx_a)  # 计算标准差的宽度
    #计算均值
    ele_all = []
    sum = 0 
    ele_sum = 0
    for index in range(len(hist)):
        sum += index * hist[index] # 计算所有司昂宿点的和
        ele_sum += hist[index] # 统计有多少个像素点
        for i in range(int(hist[index])):
            ele_all.append(index) 
    average = sum / ele_sum # 计算均值
    #计算去除开头与结尾的均值（5%）
    few = int(ele_sum*0.05) #除去开头与结尾像素点的数量
    few_sum = 0
    few_ele_sum = 0
    for few_index in range(len(ele_all)):
        if few_index < few or few_index > len(ele_all) - few: # 如果是开头或者结尾的像素
            continue # 跳出本次循环
        few_ele_sum += 1 # 如果不是开头或者结尾，计算像素的数量
        few_sum += ele_all[few_index] # 计算所有像素点的和
    few_average = few_sum / few_ele_sum # 计算除去开头与结尾像素的平均值

    return max_a, stdWidth_a, average, few_average  # 返回最大值、标准差的宽度、所有像素的平均值、除去开头与结尾像素的平均值（截断均值）


'''遍历色彩空间计算特征值'''
def calcuVec(colorSpace, img, mask):
    vecstr = '' # 定义一个空字符串
    for cs in colorSpace: # 遍历颜色空间
        color = cv2.cvtColor(img, cs) # 将图像分别转换为Lab、HLS、RGB格式
        for i in range(3):  # 遍历各通道
            hist = cv2.calcHist([color], [i], mask, [256], [0, 255]).reshape(-1)  # 计算各通道的直方图
            vec = getStatistics(hist)  # 调用定义的getStatistics()函数，根据直方图，计算最大值（峰值）、标准差宽度、均值、截断均值
            vecstr += ',' # 用‘,’隔开字符串
            vecstr += str(vec[0][0]) # 取出最大值转换为字符串类型，并添加到vecstr后面
            vecstr += ',' # 用‘,’隔开字符串
            vecstr += str(vec[1][0]) # 取出标准差宽度转换为字符串类型，并添加到vecstr后面
            vecstr += ',' # 用‘,’隔开字符串
            vecstr += str(vec[2]) # 取出均值转换为字符串类型，并添加到vecstr后面
            vecstr += ',' # 用‘,’隔开字符串
            vecstr += str(vec[3]) # 取出截断均值转换为字符串类型，并添加到vecstr后面
    return vecstr

'''获得特征向量'''
def getVec(imgPath): 
    colorSpace = [cv2.COLOR_BGR2Lab, cv2.COLOR_BGR2HLS, cv2.COLOR_BGR2RGB] # 构造色彩空间，cv2.COLOR_BGR2Lab：将BGR格式的图像转换为Lab格式；cv2.COLOR_BGR2HLS：将BGR格式的图像转换为HLS格式；cv2.COLOR_BGR2RGB：将BGR格式的图像转换为RGB格式
    img = cv2.imread(imgPath) # 读取图像
    vecStr = '' # 定义一个空字符串
    # 调用定义的seg_tongue()方法获取心肺、肝（左）、肝（右）、肾、脾对应的区域以及舌体mask
    maskOnlyTongue, kidney_mask_res, lung_mask_res, spleen_mask_res, liver_left_mask_res, liver_right_mask_res = segmentation_tongue.seg_tongue(
        imgPath,
        hist=True)
    if list(maskOnlyTongue) == [0]: 
        return vecStr, [0, 0, 0, 0, 0]

    #统计各区域的像素点数量以及占总量的比值
    sum_kidney = np.sum([kidney_mask_res == 255]) # 统计肾区域的像素数量
    sum_lung = np.sum([lung_mask_res == 255]) # 统计肺区域的像素数量
    sum_spleen = np.sum([spleen_mask_res == 255]) # 统计脾区域像素的数量
    sum_liver_left = np.sum([liver_left_mask_res == 255]) # 统计左肝区域像素的数量
    sum_liver_right = np.sum([liver_right_mask_res == 255]) # 统计右肝区域像素的数量
    all_sum = sum_kidney + sum_lung + sum_spleen + sum_liver_left + sum_liver_right # 统计五个区域像素数量的总和
    scale_kidney = sum_kidney / all_sum # 计算肾区域的占比
    scale_lung = sum_lung / all_sum # 计算肺区域的占比
    scale_spleen = sum_spleen / all_sum # 计算脾区域的占比
    scale_liver_left = sum_liver_left / all_sum # 计算左肝区域的占比
    scale_liver_right = sum_liver_right / all_sum # 计算右肝区域的占比

    tongue = calcuVec(colorSpace, img, maskOnlyTongue) # 调用定义的calcuVec()函数，计算舌体的特征值
    vecStr_kidney = calcuVec(colorSpace, img, kidney_mask_res) # 调用定义的calcuVec()函数，计算肾的特征值
    vecStr += vecStr_kidney 
    tongue_kidney = vecStr_kidney
    vecStr_lung = calcuVec(colorSpace, img, lung_mask_res) # 调用定义的calcuVec()函数，计算肺的特征值
    vecStr += vecStr_lung
    tongue_lung = vecStr_lung
    vecStr_spleen = calcuVec(colorSpace, img, spleen_mask_res) # 调用定义的calcuVec()函数，计算脾的特征值
    vecStr += vecStr_spleen
    tongue_spleen = vecStr_spleen
    vecStr_liver_left = calcuVec(colorSpace, img, liver_left_mask_res) # 调用定义的calcuVec()函数，计算左肝的特征值
    vecStr_liver_right = calcuVec(colorSpace, img, liver_right_mask_res) # 调用定义的calcuVec()函数，计算右肝的特征值
    vecStr += vecStr_liver_left
    vecStr += vecStr_liver_right
    tongue_liver_left = vecStr_liver_left
    tongue_liver_right = vecStr_liver_right
    tongue_features = [tongue, tongue_kidney, tongue_lung, tongue_spleen, tongue_liver_left, tongue_liver_right]

    return vecStr, [scale_kidney, scale_lung, scale_spleen, scale_liver_left, scale_liver_right], tongue_features