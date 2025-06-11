import cv2
import numpy as np
from math import sqrt
from skimage import measure

'''获取舌头的最小正外接矩形'''


def get_tongue(img):
    min_x = 0  # 初始化左上角点的横坐标值
    min_y = 0  # 初始化左上角点的纵坐标值
    max_x = 0  # 初始化右下角点的横坐标值
    max_y = 0  # 初始化右下角点的纵坐标值
    for x in range(len(img)):  # 按行遍历分割图像
        x_sum = np.sum(img[x])  # 计算当前行像素值的和
        if x == 0:  # 如果是第一行
            if x_sum > 0:  # 如果第一行的像素值的和大于0
                min_y = 0  # 则左上角点的纵坐标为0
            continue  # 跳出本次循环
        if x == len(img) - 1:  # 如果是最后一行
            if x_sum > 0:  # 如果最后一行的像素值的和大于0
                max_y = x  # 则右下角点的纵坐标为len(img) -1
        if np.sum(img[x - 1]) == 0 and np.sum(img[x]) > 0:  # 如果前一行的像素值的和为0，且当前行的像素值的和大于0
            min_y = x  # 则左上角点的纵坐标为x
        if np.sum(img[x - 1]) > 0 and np.sum(img[x]) == 0:  # 如果前一行的像素值的和大于0，且当前行的像素值的和等于0
            max_y = x  # 则右下角点的纵坐标为x
    for y in range(len(img[0])):  # 按列遍历分割图像
        y_sum = np.sum(img[:, y])  # 计算当前列像素和的值
        if y == 0:  # 如果是第一列
            if y_sum > 0:  # 如果第一列的像素值的和大于0
                min_x = y  # 则左上角点的横坐标为0
            continue  # 跳出本次循环
        if y == len(img[0]) - 1:  # 如果是最后一列
            if y_sum > 0:  # 如果最后一列像素值的和大于0
                max_x = y  # 则右下角点的横坐标为len(img[0]) - 1
        if np.sum(img[:, y - 1]) == 0 and np.sum(img[:, y]) > 0:  # 如果前一列的像素值的和为0，且当前列的像素值的和大于0
            min_x = y  # 则左上角点的横坐标为y
        if np.sum(img[:, y - 1]) > 0 and np.sum(img[:, y]) == 0:  # 如果前一列的像素值的和大于0，且当前列的像素值的和等于0
            max_x = y  # 则右下角点的横坐标为y
    return min_x, max_x, min_y, max_y  # 返回左上角和右下角两个点的坐标


'''根据输入的贝塞尔的控制点坐标，输出光滑曲线。'''


class Bezier2(object):
    def __init__(self, points):
        self.points = points

    def moveto(self, p):
        self.current_point = p

    def render(self):
        NO = 3
        KT = 5
        m = NO - 1
        p = {}  # p[3][2]
        for i in range(0, NO, 2):
            p[i] = self.points[i]

        l1 = 1.0 * (self.points[0][0] - self.points[1][0])
        ll = 1.0 * (self.points[0][1] - self.points[1][1])
        l1 = sqrt(l1 * l1 + ll * ll)
        l2 = 1.0 * (self.points[2][0] - self.points[1][0])
        ll = 1.0 * (self.points[2][1] - self.points[1][1])
        l2 = sqrt(l2 * l2 + ll * ll)
        p[1] = (
            ((l1 + l2) * (l1 + l2) * self.points[1][0] - l2 * l2 * self.points[0][0] - l1 * l1 * self.points[2][0]) / (
                    2 * l1 * l2),
            ((l1 + l2) * (l1 + l2) * self.points[1][1] - l2 * l2 * self.points[0][1] - l1 * l1 * self.points[2][1]) / (
                    2 * l1 * l2)
        )
        pk = {}
        for i in range(m + 1):
            pk[i] = p[i]
        pt = {}
        for k in range(KT + 1):
            for i in range(0, m + 1, 2):
                pt[2 * i] = pk[i]
            for i in range(m):
                pt[2 * i + 1] = (
                    int(pk[i][0] + pk[i + 1][0]) >> 1,
                    int(pk[i][1] + pk[i + 1][1]) >> 1
                )
            for i in range(1, m):
                pt[2 * i] = (
                    int(pt[2 * i - 1][0] + pt[2 * i + 1][0]) >> 1,
                    int(pt[2 * i - 1][1] + pt[2 * i + 1][1]) >> 1
                )
            for i in range(2 * m + 1):
                pk[i] = pt[i]

            if k == KT:
                break
            m <<= 1
        self.moveto(pk[0])
        res = []
        for i in range(1, 2 * m + 1):
            res.append(pk[i])
        return res


'''脏器分割'''


def viscera_split(mask):
    img = mask.copy()  # 获取分割结果图像
    min_x, max_x, min_y, max_y = get_tongue(img)  # 获取舌头的最小正外接矩形的左上角和右下角两个点的坐标
    tongue_h = max_y - min_y  # 计算最小正外接矩形的高

    # 肾线
    kidney_line_y = int(tongue_h * 0.4) + min_y  # 计算肾区域的高
    kidney_line_x_left = 0  # 初始化肾线左端点的横坐标
    kidney_line_x_right = 0  # 初始化肾线右端点的横坐标
    kidney_line_all = img[kidney_line_y]  # 取出kidney_line_y个点
    for i in range(len(kidney_line_all)):  # 遍历取出的kidney_line_y个点
        if i == 0:  # 如果是取出的第一个像素点
            if kidney_line_all[i] > 0:  # 且如果像素值大于0
                kidney_line_x_left = i  # 那么肾区域最左边的横坐标为0
            continue  # 跳出本次循环
        if i == len(kidney_line_all) - 1:  # 如果是取出的最后一个像素点
            if kidney_line_all[i] > 0:  # 且如果像素值大于0
                kidney_line_x_right = i  # 那么肾区域最右边的横坐标为len(kidney_line_all) - 1
        if kidney_line_all[i - 1] == 0 and kidney_line_all[i] > 0:  # 如果索引为i点的前一个点的像素值为0，且i点的像素值大于0
            kidney_line_x_left = i  # 那么肾线左端点的横坐标为i
        if kidney_line_all[i - 1] > 0 and kidney_line_all[i] == 0:  # 如果索引为i点的前一个点的像素值大于0，且i点的像素值等于0
            kidney_line_x_right = i  # 那么肾线右端点的横坐标为i
    kidney_line_dot_left = (kidney_line_x_left - 2, kidney_line_y)  # 肾线左端点的坐标
    kidney_line_dot_right = (kidney_line_x_right + 2, kidney_line_y)  # 肾线右端点的坐标

    # 肺线
    lung_line_y = int(tongue_h * 0.8) + min_y  # 计算肺区域的高
    lung_line_x_left = 0  # 初始化肺线左端点的横坐标
    lung_line_x_right = 0  # 初始化肺线右端点的横坐标
    lung_line_all = img[lung_line_y]  # 取出lung_line_y个像素点
    for i in range(len(lung_line_all)):  # 遍历取出的lung_line_y个像素点
        if i == 0:  # 如果是取出的第一个像素点
            if lung_line_all[i] > 0:  # 且如果像素值大于0
                lung_line_x_left = i  # 那么肺区域最左边的横坐标为0
            continue  # 跳出本次循环
        if i == len(lung_line_all) - 1:  # 如果是取出的最后一个像素点
            if lung_line_all[i] > 0:  # 且如果像素值大于0
                lung_line_x_right = i  # 那么肾区域最右边的横坐标为len(kidney_line_all) - 1

        if lung_line_all[i - 1] == 0 and lung_line_all[i] > 0:  # 如果索引为i点的前一个点的像素值为0，且i点的像素值大于0
            lung_line_x_left = i  # 那么肺区域最左边的横坐标为i
        if lung_line_all[i - 1] > 0 and lung_line_all[i] == 0:  # 如果索引为i点的前一个点的像素值大于0，且i点的像素值等于0
            lung_line_x_right = i  # 那么肺区域最右边的横坐标为i
    lung_line_dot_left = (lung_line_x_left - 2, lung_line_y)  # 肺线左端点的坐标
    lung_line_dot_right = (lung_line_x_right + 2, lung_line_y)  # 肺线右端点的坐标

    # 脾线
    spleen_line_y = int((lung_line_y - kidney_line_y) / 2 + kidney_line_y)  # 找到脾线的纵坐标
    spleen_line_x_left = 0  # 初始化脾线左端点的横坐标
    spleen_line_x_right = 0  # 初始化脾线右端点的横坐标
    spleen_line_all = img[spleen_line_y]  # 取出spleen_line_y个像素点
    for i in range(len(spleen_line_all)):  # 遍历所有取出的像素点
        if i == 0:  # 如果是取出的第一个像素点
            if spleen_line_all[i] > 0:  # 且如果像素值大于0
                spleen_line_x_left = i  # 那么脾线左端点的横坐标为0
            continue  # 跳出本次循环
        if i == len(spleen_line_all) - 1:  # 如果是取出的最后一个像素点
            if spleen_line_all[i] > 0:  # 且如果像素值大于0
                spleen_line_x_right = i  # 那么脾线右端点的横坐标为len(spleen_line_all) - 1
        if spleen_line_all[i - 1] == 0 and spleen_line_all[i] > 0:  # 如果索引为i点的前一个点的像素值为0，且i点的像素值大于0
            spleen_line_x_left = i  # 那么脾线左端点的横坐标为i
        if spleen_line_all[i - 1] > 0 and spleen_line_all[i] == 0:  # 如果索引为i点的前一个点的像素值大于0，且i点的像素值等于0
            spleen_line_x_right = i  # 那么脾线右端点的横坐标为i

    new_spleen_line_x_left = spleen_line_x_left + (spleen_line_x_right - spleen_line_x_left) * 0.2  # 确定最终脾线左端点的横坐标
    new_spleen_line_x_right = spleen_line_x_left + (spleen_line_x_right - spleen_line_x_left) * 0.8  # 确定最终脾线右端点的横坐标

    spleen_dot_left = (int(new_spleen_line_x_left), spleen_line_y)  # 脾线左端点的坐标
    spleen_dot_right = (int(new_spleen_line_x_right), spleen_line_y)  # 脾线右端点的坐标

    kidney_line_mid = (
    int((kidney_line_x_right - kidney_line_x_left) / 2 + kidney_line_x_left), kidney_line_y - 10)  # 肾线中间点
    lung_line_mid = (int((lung_line_x_right - lung_line_x_left) / 2 + lung_line_x_left), lung_line_y - 10)  # 肺线中间点

    dot_list = []  # 初始化空列表
    b1 = Bezier2((kidney_line_dot_left, spleen_dot_left, lung_line_dot_left))  # 利用肾线左端点，脾线左端点，肺线左端点三个点绘制二阶贝塞尔曲线
    b2 = Bezier2((kidney_line_dot_right, spleen_dot_right, lung_line_dot_right))  # 利用肾线右端点，脾线右端点，肺线右端点三个点绘制二阶贝塞尔曲线
    b3 = Bezier2((kidney_line_dot_left, kidney_line_mid, kidney_line_dot_right))  # 利用肾线左端点，肾线中间点，肾线右端点绘制二阶贝塞尔曲线
    b4 = Bezier2((lung_line_dot_left, lung_line_mid, lung_line_dot_right))  # 利用肺线左端点，肺线中间点，肺线右端点绘制二阶贝塞尔曲线
    # 将二阶贝塞尔曲线各个点的坐标存储到列表中
    dot_list.append(b1.render())
    dot_list.append(b2.render())
    dot_list.append(b3.render())
    dot_list.append(b4.render())
    # 画线，后面会根据线寻找连通区域
    for i in dot_list:
        for index in range(len(i)):
            if index == 0:
                continue
            cv2.line(mask, i[index - 1], i[index], (0, 0, 0), 3, 4)  # 根据坐标绘制曲线
    mask[mask == 1] = 255  # 将属于舌体的像素值设置为255
    cv2.imwrite('mask.jpg', mask)

    # 寻找连通区域
    from skimage import measure
    labeled_img, num = measure.label(mask, background=0, return_num=True, connectivity=2)  # 调用 measure.label方法标记最大连通区域
    class_dict = {}
    for i in range(len(labeled_img)):
        for j in range(len(labeled_img[i])):
            if labeled_img[i][j] == 0:
                continue
            key = labeled_img[i][j]
            if key not in class_dict.keys():
                class_dict[key] = [(i, j)]
            else:
                class_dict[key].append((i, j))

    if num < 5:  # 如果连通区域的数量小于5，则舌体脏腑对应部位分割失败，请重新拍摄
        print('舌体脏腑对应部位分割失败，请重新拍摄')
        return [0], [0], [0], [0], [0]
    if num > 5:
        while True:
            if len(class_dict.keys()) == 5:
                break
            min_key = 0
            min_len = labeled_img.shape[0] * labeled_img.shape[1]
            for key in class_dict.keys():
                if len(class_dict[key]) < min_len:
                    min_key = key
                    min_len = len(class_dict[key])
            del class_dict[min_key]

    kidney_mask = []
    lung_mask = []
    spleen_mask = []
    liver_left_mask = []
    liver_right_mask = []

    kidney = [0, [labeled_img.shape[0], 0]]
    lung = [0, [0, 0]]
    liver_left = [0, [0, labeled_img.shape[1]]]
    liver_right = [0, [0, 0]]
    # 确定各个连通区域属于的部位
    for key in class_dict.keys():
        centre = (0, 0)
        for i in class_dict[key]:
            centre = (centre[0] + i[0], centre[1] + i[1])
        centre = [int(centre[0] / len(class_dict[key])), int(centre[1] / len(class_dict[key]))]
        if centre[0] < kidney[1][0]:
            kidney = (key, centre)
        if centre[0] > lung[1][0]:
            lung = (key, centre)
        if centre[1] < liver_left[1][1]:
            liver_left = (key, centre)
        if centre[1] > liver_right[1][1]:
            liver_right = (key, centre)
    # 若有一个连通区域未找到则返回空
    if kidney[0] == 0 or lung[0] == 0 or liver_left[0] == 0 or liver_right[0] == 0:
        return [0], [0], [0], [0], [0]
    kidney_mask.extend(class_dict[kidney[0]])
    lung_mask.extend(class_dict[lung[0]])
    liver_left_mask.extend(class_dict[liver_left[0]])
    liver_right_mask.extend(class_dict[liver_right[0]])
    # 删除四个区域剩下的是胃
    if kidney[0] in class_dict.keys():
        del class_dict[kidney[0]]
    if lung[0] in class_dict.keys():
        del class_dict[lung[0]]
    if liver_left[0] in class_dict.keys():
        del class_dict[liver_left[0]]
    if liver_right[0] in class_dict.keys():
        del class_dict[liver_right[0]]
    for key in class_dict.keys():
        spleen_mask.extend(class_dict[key])

    return kidney_mask, lung_mask, spleen_mask, liver_left_mask, liver_right_mask
