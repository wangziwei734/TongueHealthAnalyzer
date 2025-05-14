from analysis.merge_features import *
import pandas as pd
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.curdir))

def Main(user_data, region_rate, user_history_healthy=None):

    ls = pd.read_csv(root_path + '/LS.csv', encoding='utf-8', header=None) # 加载经验权重
    ls = np.array(ls.loc[0, 180:]) # 取180~359列的数据
    user_data = np.array(user_data) # 将用户各区域的特征向量转换为数组类型

    user_region_score = [] # 每个用户每一次检测的每个区域的特征融合结果
    for j in range(0, user_data.shape[0], 36): # 遍历用户的特征向量
        region = user_data[j:j + 36] # 选取一个区域的特征向量，每个区域有36个特征
        region_ls = ls[j:j + 36] # 选取一个区域的经验权重
        tmp = merge_feature(region, region_ls) # 调用定义的merge_feature()方法，融合特征
        user_region_score.append(tmp) # 将每个区域的融合后的特征保存到数组中

    scaler_user_tongue_score = scaler_feature(user_region_score) # 调用定义的scaler_feature()方法，归一化处理区域得分
    merge_tongue_score = merge_region(scaler_user_tongue_score, region_rate) # 按照区域占比，融合所有区域的得分
    # 提取心、肝、脾、肺、肾五个区域的得分
    merge_heart_score = scaler_user_tongue_score[1] # 心
    merge_spleen_score = scaler_user_tongue_score[2] # 脾
    merge_kidney_score = scaler_user_tongue_score[0] # 肾
    merge_lung_score = scaler_user_tongue_score[1] # 肺
    # 合并左肝、右肝
    scaler_user_tongue_liver_score = (scaler_user_tongue_score[3] + scaler_user_tongue_score[4]) / 2
    merge_liver_score = scaler_user_tongue_liver_score # 肝

    if user_history_healthy is not None and len(user_history_healthy.shape) > 1: # 如果用户历史健康得分不为空且是二维数组
        user_heart_score = healthy_score(merge_heart_score, user_history_healthy[:, 1]) # 将心的健康值得分映射到[-1, 1]之间
        user_spleen_score = healthy_score(merge_spleen_score, user_history_healthy[:, 2]) # 将脾的健康值得分映射到[-1, 1]之间
        user_kidney_score = healthy_score(merge_kidney_score, user_history_healthy[:, 3]) # 将肾的健康值得分映射到[-1, 1]之间
        user_lung_score = healthy_score(merge_lung_score, user_history_healthy[:, 4]) # 将肺的健康值得分映射到[-1, 1]之间
        user_liver_score = healthy_score(merge_liver_score, user_history_healthy[:, 5]) # 将肝的健康值得分映射到[-1, 1]之间
        user_healthy_score = healthy_score(merge_tongue_score, user_history_healthy[:, 0]) # 将整体健康值得分映射到[-1, 1]之间
        hy = [user_healthy_score[-1], user_heart_score[-1], user_spleen_score[-1], user_kidney_score[-1],
              user_lung_score[-1], user_liver_score[-1]]
    else: # 如果用户历史健康得分为空或是一维数组，预设用户各脏器的健康值为0，即预设用户是健康的
        user_heart_score = healthy_score(merge_heart_score, [0])
        user_spleen_score = healthy_score(merge_spleen_score, [0])
        user_kidney_score = healthy_score(merge_kidney_score, [0])
        user_lung_score = healthy_score(merge_lung_score, [0])
        user_liver_score = healthy_score(merge_liver_score, [0])
        user_healthy_score = healthy_score(merge_tongue_score, [0])
        hy = [float(user_healthy_score[-1]), float(user_heart_score[-1]), float(user_spleen_score[-1]),
              float(user_kidney_score[-1]), float(user_lung_score[-1]), float(user_liver_score[-1])]
    return hy