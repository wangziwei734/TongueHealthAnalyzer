from flask import Flask, request, jsonify
import os
import time
import base64
import uuid
import cv2
import numpy as np
from flask_cors import CORS
import ChineseMedicine_analysis

app = Flask(__name__)
# 配置CORS，允许来自TongueScanPro域名的请求
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://www.tonguescanpro.com",
    "http://tonguescanpro.com",
    "http://api.tonguescanpro.com"
]}})

# 根目录路径
root_path = os.path.abspath(os.path.dirname(__file__))
upload_folder = os.path.join(root_path, 'upload')
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

def rm_img(path):
    """删除文件"""
    if os.path.exists(path):
        os.remove(path)

def save_base64_img(base64_data, store_path):
    """保存Base64图像数据"""
    try:
        # 移除Base64前缀(如果有)
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        # 解码Base64数据
        img_data = base64.b64decode(base64_data)
        
        # 将二进制数据转换为numpy数组
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 检查解码是否成功
        if img is None:
            return False, "Base64解码失败"
        
        # 保存图像
        img_shape = img.shape
        rate = max(img.shape[0]//1000, img.shape[1]//500)
        if rate > 1:
            img = cv2.resize(img, (img.shape[1]//rate, img.shape[0]//rate))
        
        cv2.imwrite(store_path, img)
        return True, rate
    except Exception as e:
        return False, str(e)

def analysis(img_path, user_id):
    """对舌部图像进行分析"""
    try:
        time_start = time.time()
        health_value = ChineseMedicine_analysis.analysis_ChineseMedicine(img_path, user_id)
        time_end = time.time()
        
        res = {
            'healthy': float(health_value[0]),  # 整体健康值
            'heart': float(health_value[1]),    # 心健康值
            'spleen': float(health_value[2]),   # 脾健康值
            'kidney': float(health_value[3]),   # 肾健康值
            'lung': float(health_value[4]),     # 肺健康值
            'liver': float(health_value[5])     # 肝健康值
        }
        
        print(f"中医分析成功，耗时: {time_end - time_start:.2f}s")
        return res
    except Exception as error:
        print(f"中医分析失败: {error}")
        return None

def get_diagnosis(health_result):
    """根据健康值生成诊断结果"""
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
    
    return res

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查API"""
    return jsonify({"status": "ok", "message": "服务正常运行"})

@app.route('/api/analyze', methods=['POST'])
def analyze_tongue():
    """舌诊分析API"""
    # 获取请求数据
    data = request.json
    
    if not data or 'image' not in data:
        return jsonify({
            "success": False,
            "message": "请求数据不完整，缺少图像数据"
        }), 400
    
    # 从请求中获取图像数据和用户ID
    base64_img = data['image']
    user_id = data.get('userId', str(uuid.uuid4()))  # 如果没有提供用户ID，则生成一个随机ID
    
    # 生成临时文件名
    img_name = f"{str(time.time()).replace('.', '')}_{user_id}.jpg"
    img_path = os.path.join(upload_folder, img_name)
    
    # 保存图像
    success, result = save_base64_img(base64_img, img_path)
    if not success:
        return jsonify({
            "success": False,
            "message": f"图像保存失败: {result}"
        }), 400
    
    # 分析图像
    try:
        health_result = analysis(img_path, user_id)
        
        # 分析完成后删除临时文件
        rm_img(img_path)
        
        if health_result is None:
            return jsonify({
                "success": False,
                "message": "舌诊分析失败"
            }), 500
        
        # 获取诊断结果
        diagnosis = get_diagnosis(health_result)
        
        # 返回结果
        return jsonify({
            "success": True,
            "userId": user_id,
            "results": health_result,
            "diagnosis": diagnosis
        })
        
    except Exception as e:
        # 确保删除临时文件
        rm_img(img_path)
        return jsonify({
            "success": False,
            "message": f"分析过程中出错: {str(e)}"
        }), 500

if __name__ == '__main__':
    # 预热模型
    print("正在加载模型...")
    ChineseMedicine_analysis.get_model()
    print("模型加载完成，API服务启动...")
    
    # 启动服务
    app.run(host='0.0.0.0', port=5000, debug=False) 