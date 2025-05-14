import sys
import os
import tensorflow as tf
from keras_segmentation.models.unet import resnet50_unet
import glob
import numpy as np

# 禁用eager execution
tf.compat.v1.disable_eager_execution()

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# h5格式权重路径
weights_dir = os.path.join(current_dir, 'weights')

# 查找并获取轮次最高的h5权重文件 - 更高效的版本
def get_latest_h5_weights():
    # 直接获取目录中的所有h5文件
    h5_files = [f for f in os.listdir(weights_dir) if f.endswith('.h5') and f.startswith('resunet.')]
    
    if not h5_files:
        return None
    
    # 提取数字部分并排序
    max_epoch = -1
    max_file = None
    
    for file_name in h5_files:
        # 从文件名中提取数字 (resunet.X.h5 格式)
        parts = file_name.split('.')
        if len(parts) >= 3 and parts[0] == 'resunet':
            try:
                epoch = int(parts[1])
                if epoch > max_epoch:
                    max_epoch = epoch
                    max_file = os.path.join(weights_dir, file_name)
            except ValueError:
                continue
    
    return max_file

# 根据需要加载模型的全局变量
MODEL = None

# 获取或加载模型
def get_model():
    global MODEL
    if MODEL is not None:
        return MODEL
        
    # 获取最新的h5权重文件
    latest_h5_weights = get_latest_h5_weights()
    
    # 加载h5权重模型
    if latest_h5_weights and os.path.exists(latest_h5_weights):
        print(f"正在加载最新h5格式权重: {latest_h5_weights}")
        # 创建模型并加载权重
        MODEL = resnet50_unet(n_classes=2, input_height=576, input_width=768)
        MODEL.load_weights(latest_h5_weights)
        print("✓ 成功加载h5格式权重")
    else:
        print("未找到h5格式权重文件")
        MODEL = None
        
    return MODEL

# 预加载模型
model = get_model()

# 导入其他必要模块，排除可能有问题的依赖
# from faceDetector import face_detector_dlib
# from tongueDetector import tongue_detector_dlib
from tongue.segmentation_tongue import seg_tongue
from tongue import tongueHist
from analysis.main import Main

root_path = os.path.abspath(os.path.dirname(__file__)) 
features_path = os.path.join(root_path, 'features') 

def analysis_ChineseMedicine(upl_tongue_path, user_id):
    """
    分析舌体图像，计算健康值
    
    参数:
    upl_tongue_path: 舌体图像路径
    user_id: 用户ID
    
    返回:
    health_value: 健康值数组 [整体, 心, 脾, 肾, 肺, 肝]
    """
    tongue_features, scale_tongue, tongue_features_all = tongueHist.getVec(upl_tongue_path) # 调用定义的getVec()方法，获取舌体的特征值、各区域所占比值、各区域的特征值
    history_id_path = features_path + '/' + user_id #某用户健康值保存的路径
    if not os.path.exists(history_id_path): # 如果不存在该用户对应的目录
        os.makedirs(history_id_path) # 则为该用户创建一个目录
    features_id_list = os.listdir(history_id_path) # 返回history_id_path目录下的文件列表
    if len(features_id_list) == 0: # 如果列表为空，即history_id_path路径下不存在文件
        user_history_healthy = None # 则设置user_history_healthy参数为none
    else: # 如果history_id_path路径下存在文件
        history = open(history_id_path + '/' + 'value.txt', 'r', encoding='utf-8') # 读取'value.txt'文件中的内容
        history_list = history.readlines() # 以列表的形式返回'value.txt'文件中的所有行
        history_list = [i.strip().split(',') for i in history_list] # value.txt文件中的每一行数据以','作为分割字符，默认删除空白符
        user_history_healthy = np.array(history_list, dtype=float) # 将history_list列表转换为浮点型的数组
        history.close() # 关闭'value.txt'文件
    features_all = tongue_features[1:].split(',') # 将舌体特征值组成的字符串用','分割成列表
    features_all = np.array(features_all, dtype=float) # 将处理好的features_all列表转换为浮点型的数组
    health_value = Main(features_all, scale_tongue, user_history_healthy=user_history_healthy) # 调用定义的Main函数，计算各区域以及整体的健康值
    history_w = open(history_id_path + '/' + 'value.txt', 'a+', encoding='utf-8') # 打开'value.txt'文件
    history_w_value = str(health_value[0]) + ',' + str(health_value[1]) + ',' + str(health_value[2]) + ',' + \
                      str(health_value[3]) + ',' + str(health_value[4]) + ',' + str(health_value[5]) + '\n'
    history_w.write(history_w_value) # 将计算的各区域的健康值写入value.txt文件
    history_w.close() # 关闭value.txt文件
    return health_value
    
if __name__ == "__main__":
    # 测试功能
    if len(sys.argv) > 2:
        image_path = sys.argv[1]
        user_id = sys.argv[2]
    else:
        image_path = 'example/2.jpg'  # 默认测试图像
        user_id = 'test_user'  # 默认用户ID
        
    print(f"分析图像: {image_path}")
    print(f"用户ID: {user_id}")
    
    try:
        health_value = analysis_ChineseMedicine(image_path, user_id)
        print("\n健康分析结果:")
        print(f"整体健康值: {health_value[0]:.4f}")
        print(f"心健康值: {health_value[1]:.4f}")
        print(f"脾健康值: {health_value[2]:.4f}")
        print(f"肾健康值: {health_value[3]:.4f}")
        print(f"肺健康值: {health_value[4]:.4f}")
        print(f"肝健康值: {health_value[5]:.4f}")
        
        # 输出体质判断
        res = []
        if abs(health_value[1]) > 0.4:
            res.append('血虚')
        if abs(health_value[2]) > 0.25:
            res.append('脾虚')
        if abs(health_value[3]) > 0.4:
            res.append('肾虚')
        if abs(health_value[4]) > 0.4:
            res.append('气虚')
        if abs(health_value[5]) > 0.45:
            res.append('肝郁')
        if not res:
            res.append('健康')
            
        print(f"\n体质判断: {', '.join(res)}")
    except Exception as e:
        print(f"分析失败: {e}")