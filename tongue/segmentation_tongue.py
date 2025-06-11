import os
from keras_segmentation.predict import model_from_checkpoint_path, predict
from tongue.tongue_segmentation import segmentation
import cv2
import numpy as np
import tensorflow as tf
import glob

# 获取当前文件所在目录的父目录（项目根目录）
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 禁用eager execution
tf.compat.v1.disable_eager_execution()

# 替换 get_default_graph()
graph = tf.compat.v1.get_default_graph()

# 使用ChineseMedicine_analysis中的模型
try:
    # 导入模型，只有在使用时才导入，避免循环导入
    from ChineseMedicine_analysis import get_model

    model = get_model()
    if model is None:
        raise Exception("未能从ChineseMedicine_analysis获取模型")
except Exception as e:
    print(f"从ChineseMedicine_analysis导入模型失败: {e}")
    raise Exception("未找到可用的权重文件")

'''舌面分割,若hist为true则返回各五脏对应区域mask'''


def seg_tongue(img_input, hist=False):
    global graph
    with graph.as_default():  # as_default的操作是将新建的图在当前这个with语句块的范围内设置为默认图
        pr, img = predict(model=model, inp=img_input)  # 获取预测结果
    img = np.array(img[:, :, 0].copy(), dtype=np.uint8)  # 取图像B通道的像素值，并生成uint8类型的数组

    # 获取最大连通域
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)  # 使用使用cv2.findContours()函数来查找舌体的轮廓，并返回轮廓一系列的坐标值
    maxSize = 0  # 初始化最大轮廓面积的编号，即最大连通区域的编号
    for i in range(len(contours)):
        if cv2.contourArea(contours[maxSize]) < cv2.contourArea(
                contours[i]):  # 使用cv2.contourArea()函数计算轮廓的面积，如果轮廓maxSize的面积小于轮廓i的面积
            maxSize = i  # 则更新maxSize的值
    mask = img.copy()  # 定义mask
    mask[:, :] = 0  # 将mask上的所有像素赋值为0
    cv2.drawContours(mask, contours, maxSize, (255, 255, 255), -1)  # 使用cv2.drawContours()函数绘制轮廓
    maskOnlyTongue = mask.copy()  # 得到舌体区域

    # 获取mask
    if hist:
        kidney_mask, lung_mask, spleen_mask, liver_left_mask, liver_right_mask = segmentation.viscera_split(
            mask)  # 调用定义的viscera_split()函数，将舌体划分为各脏器对应的区域
        if kidney_mask == [0]:  # 如果肾区域为空，则所有区域返回空
            return [0], [0], [0], [0], [0], [0],
        # 划分出肾对应的区域
        kidney_mask_res = img.copy()
        kidney_mask_res[:, :] = 0
        for i in kidney_mask:
            kidney_mask_res[i[0]][i[1]] = 255
            # 划分出心对应的区域
        lung_mask_res = img.copy()
        lung_mask_res[:, :] = 0
        for i in lung_mask:
            lung_mask_res[i[0]][i[1]] = 255
        # 划分出脾对应的区域
        spleen_mask_res = img.copy()
        spleen_mask_res[:, :] = 0
        for i in spleen_mask:
            spleen_mask_res[i[0]][i[1]] = 255
        # 划分出肝对应的舌体左侧区域
        liver_left_mask_res = img.copy()
        liver_left_mask_res[:, :] = 0
        for i in liver_left_mask:
            liver_left_mask_res[i[0]][i[1]] = 255
        # 划分出肝对应的舌体右侧区域
        liver_right_mask_res = img.copy()
        liver_right_mask_res[:, :] = 0
        for i in liver_right_mask:
            liver_right_mask_res[i[0]][i[1]] = 255
        return maskOnlyTongue, kidney_mask_res, lung_mask_res, spleen_mask_res, liver_left_mask_res, liver_right_mask_res

    return maskOnlyTongue
