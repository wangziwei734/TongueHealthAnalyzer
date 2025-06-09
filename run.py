import os
import time
import traceback
import ChineseMedicine_analysis
import cv2
import numpy as np

# 根目录路径
root_path = os.path.abspath(os.path.dirname(__file__))
upload_folder = os.path.join(root_path, 'upload')
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)
    
def rm_img(path):
    """删除文件"""
    if os.path.exists(path):
        os.remove(path)
        
def save_img(file_path, store_path):
    """resize并保存数据，返回缩放比率"""
    rm_img(store_path)
    img = cv2.imread(file_path)
    img_shape = img.shape
    rate = max(img.shape[0]//1000, img.shape[1]//500)
    if rate > 1:
        img = cv2.resize(img, (img.shape[1]//rate, img.shape[0]//rate))
        cv2.imwrite(store_path, img)
        return rate, img_shape
    else:
        cv2.imwrite(store_path, img)
        return 1, img_shape

def analysis(upl_tongue, user_id):
    """对舌部图像进行分析"""
    upl_tongue_name = str(time.time()).replace('.', '') + '.jpg'
    upl_tongue_path = os.path.join(upload_folder, upl_tongue_name)
    try:
        upl_tongue_rate, _ = save_img(upl_tongue, upl_tongue_path)
    except IOError:
        rm_img(upl_tongue_path)
        print("保存图像数据失败")
        return None
        
    try:
        time_start = time.time()
        health_value = ChineseMedicine_analysis.analysis_ChineseMedicine(upl_tongue_path, user_id)
        time_end = time.time()
        rm_img(upl_tongue_path)
        
        res = {
            'healthy': health_value[0],  # 整体健康值
            'heart': health_value[1],    # 心健康值
            'spleen': health_value[2],   # 脾健康值
            'kidney': health_value[3],   # 肾健康值
            'lung': health_value[4],     # 肺健康值
            'liver': health_value[5]     # 肝健康值
        }
        
        print(f"中医分析成功，耗时: {time_end - time_start:.2f}s")
        return res
    except Exception as error:
        traceback.print_exc()
        rm_img(upl_tongue_path)
        print(f"中医分析失败: {error}")
        return None

def run_analysis(image_path, user_id='test_user'):
    """执行简化版的舌体分析流程"""
    print(f"开始分析图像: {image_path}")
    
    # 舌体分析
    health_result = analysis(image_path, user_id)
    if health_result is None:
        print("舌体分析失败")
        return None
    
    # 解释结果
    res = []
    if abs(health_result['heart']) > 0.4:
        res.append('血虚')
    if abs(health_result['spleen']) > 0.25:
        res.append('脾虚')
    if abs(health_result['kidney']) > 0.4:
        res.append('肾虚')
    if abs(health_result['lung']) > 0.4:
        res.append('气虚')
    if abs(health_result['liver']) > 0.45:
        res.append('肝郁')
    if not res:
        res.append('健康')
    
    # 输出结果
    print("\n分析结果:")
    print(f"整体健康值: {health_result['healthy']:.4f}")
    print(f"心健康值: {health_result['heart']:.4f}")
    print(f"脾健康值: {health_result['spleen']:.4f}")
    print(f"肾健康值: {health_result['kidney']:.4f}")
    print(f"肺健康值: {health_result['lung']:.4f}")
    print(f"肝健康值: {health_result['liver']:.4f}")
    print(f"体质判断: {', '.join(res)}")
    
    return health_result, res

if __name__ == '__main__':
    # 默认测试图像
    image_path = 'example/5.jpg'
    
    # 使用命令行参数
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    # 运行分析
    run_analysis(image_path)