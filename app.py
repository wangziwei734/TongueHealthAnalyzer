from flask import Flask, request, jsonify
import os
import time
import base64
import uuid
import cv2
import numpy as np
from flask_cors import CORS
import tensorflow as tf
import traceback

# TensorFlow配置 - 更加彻底的TF1兼容性设置
print(f"TensorFlow版本: {tf.__version__}")

# 完全禁用TensorFlow 2.x行为
tf.compat.v1.disable_v2_behavior()
print("已禁用TensorFlow 2.x行为")

# 禁用eager execution
tf.compat.v1.disable_eager_execution()
print(f"禁用后Eager execution状态: {tf.executing_eagerly()}")

# 清理会话和默认图
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

# 创建一个全局共享的默认图
graph = tf.compat.v1.get_default_graph()

# 配置TensorFlow会话
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # 动态分配GPU内存
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 限制GPU内存使用比例

# 创建全局会话并设置为默认会话
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# 激活变量共享机制
with tf.compat.v1.variable_scope("", reuse=tf.compat.v1.AUTO_REUSE):
    pass  # 确保变量共享

# 初始化全局变量
init_op = tf.compat.v1.global_variables_initializer()
sess.run(init_op)

# 运行图初始化后再导入分析模块
import ChineseMedicine_analysis

# 根目录路径
root_path = os.path.abspath(os.path.dirname(__file__))
upload_folder = os.path.join(root_path, 'upload')
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

app = Flask(__name__)
# 配置CORS
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://www.tonguescanpro.com",
    "http://tonguescanpro.com",
    "http://api.tonguescanpro.com",
    "*"
]}})


def rm_img(path):
    """删除文件"""
    if os.path.exists(path):
        os.remove(path)


def save_img(file_path, store_path):
    """与run.py中相同的图像保存逻辑"""
    try:
        rm_img(store_path)
        img = cv2.imread(file_path)

        # 检查图像是否成功读取
        if img is None:
            print(f"无法读取图像: {file_path}")
            return None, None

        img_shape = img.shape
        rate = max(img.shape[0] // 1000, img.shape[1] // 500)
        if rate > 1:
            img = cv2.resize(img, (img.shape[1] // rate, img.shape[0] // rate))
            cv2.imwrite(store_path, img)
            return rate, img_shape
        else:
            cv2.imwrite(store_path, img)
            return 1, img_shape
    except Exception as e:
        print(f"图像处理失败: {e}")
        traceback.print_exc()
        return None, None


def save_base64_img(base64_data, final_file_path):
    """保存Base64图像数据并处理"""
    try:
        # 移除Base64前缀(如果有)
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]

        # 解码Base64数据
        try:
            img_data = base64.b64decode(base64_data)
        except Exception as e:
            print(f"Base64解码失败: {e}")
            return False, "Base64解码失败"

        # 直接从内存中解码图像
        try:
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                print("图像解码失败，无法从Base64数据创建图像")
                return False, "图像解码失败"

            # 处理图像
            img_shape = img.shape
            rate = max(img.shape[0] // 1000, img.shape[1] // 500)
            if rate > 1:
                img = cv2.resize(img, (img.shape[1] // rate, img.shape[0] // rate))

            # 保存处理后的图像
            cv2.imwrite(final_file_path, img)
            return True, rate

        except Exception as e:
            print(f"图像处理失败: {e}")
            traceback.print_exc()
            return False, f"图像处理失败: {e}"

    except Exception as e:
        print(f"图像保存流程失败: {e}")
        traceback.print_exc()
        return False, str(e)


def analysis(upl_tongue, user_id):
    """简化的舌诊分析函数，与run.py逻辑类似"""
    print(f"开始分析图像: {upl_tongue}")

    try:
        # 确保使用全局会话
        global sess, graph
        with graph.as_default():
            tf.compat.v1.keras.backend.set_session(sess)
            # 预检查变量初始化
            for var in tf.compat.v1.global_variables():
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    print(f"初始化变量: {var.name}")
                    sess.run(tf.compat.v1.variables_initializer([var]))

            # 使用同一个会话调用分析函数
            health_value = ChineseMedicine_analysis.analysis_ChineseMedicine(upl_tongue, user_id)

        if health_value is None:
            print("分析返回None结果")
            return None

        # 结果格式化
        res = {
            'healthy': health_value[0],  # 整体健康值
            'heart': health_value[1],  # 心健康值
            'spleen': health_value[2],  # 脾健康值
            'kidney': health_value[3],  # 肾健康值
            'lung': health_value[4],  # 肺健康值
            'liver': health_value[5]  # 肝健康值
        }

        print("中医分析成功")
        return res

    except Exception as error:
        print(f"中医分析失败: {error}")
        traceback.print_exc()
        return None


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

    try:
        # 生成临时文件名
        temp_name = f"temp_{str(time.time()).replace('.', '')}_{user_id}.jpg"
        temp_path = os.path.join(upload_folder, temp_name)

        # 保存Base64图像
        success, result = save_base64_img(base64_img, temp_path)

        if not success:
            return jsonify({
                "success": False,
                "message": f"图像保存失败: {result}"
            }), 400

        # 分析图像
        health_result = analysis(temp_path, user_id)

        # 确保删除临时文件
        rm_img(temp_path)

        if health_result is None:
            return jsonify({
                "success": False,
                "message": "舌诊分析失败，模型无法处理此图像"
            }), 500

        # 使用与run.py相同的阈值判断体质
        diagnosis = []
        if abs(health_result['heart']) > 0.4:
            diagnosis.append('血虚')
        if abs(health_result['spleen']) > 0.25:
            diagnosis.append('脾虚')
        if abs(health_result['kidney']) > 0.4:
            diagnosis.append('肾虚')
        if abs(health_result['lung']) > 0.4:
            diagnosis.append('气虚')
        if abs(health_result['liver']) > 0.45:
            diagnosis.append('肝郁')
        if not diagnosis:
            diagnosis.append('健康')

        # 返回结果
        return jsonify({
            "success": True,
            "userId": user_id,
            "results": health_result,
            "diagnosis": diagnosis
        })

    except Exception as e:
        # 确保删除临时文件
        if 'temp_path' in locals():
            rm_img(temp_path)
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"分析过程中出错: {str(e)}"
        }), 500


@app.route('/')
def index():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    # 启动服务前确保模型加载
    print("预加载模型...")
    model = ChineseMedicine_analysis.get_model()
    if model is None:
        print("警告: 模型加载失败")
    else:
        print("模型加载成功")

    # 启动服务
    print(f"API服务启动...")
    app.run(host='0.0.0.0', port=5000, debug=False)
