from sklearn import preprocessing
import numpy as np

def merge_feature(features, score):
    merge = np.sum(np.multiply(features, score))  
    return merge


def scaler_feature(features):
    min_max_scaler = preprocessing.MinMaxScaler() 
    features = np.array(features).reshape(-1, 1) # 将数据转换成1列
    scale = min_max_scaler.fit_transform(features)  # 将所有的特征值缩放到0~1之间
    return scale[:, 0]

def merge_region(region, rate):
    return np.sum(np.multiply(region, rate))


def healthy_score(score, history=None):
    if history is not None: # 如果映射后的得分存在
        normal = np.mean(history) # 计算平均值
        if score == normal: # 如果健康得分等于平均值
            hs = 0 # 则将健康分设置为0，表示非常健康
        elif score < normal: # 如果健康得分小于平均值
            hs = 1 - score / normal # 则将健康得分设置为1 - score / normal
        else:
            hs = -(score - normal) / (2 - normal) #如果健康得分大于平均值，重新计算健康得分
        hs = np.append(history, np.array([hs])) # 将健康得分放在映射得分后面
    return hs