# 中医舌诊系统使用指南 📋

## 项目介绍 🔍

本系统是一个基于计算机视觉和机器学习的中医舌诊分析系统，通过对舌像图片进行处理、分割和特征提取，可以自动评估用户的整体健康状况以及心、肝、脾、肺、肾五脏的健康情况，并给出相应的中医诊断建议。

### 系统特色 ✨

- **传统医学与现代技术的结合**：将传统中医舌诊理论与现代人工智能技术相结合
- **自动化舌体识别与分割**：使用深度学习模型自动检测并分割舌体区域
- **多区域健康评估**：将舌体划分为对应五脏的区域，分别进行分析
- **个性化健康记录**：记录用户历史健康数据，实现个性化分析
- **可解释性诊断结果**：不仅提供健康评分，还给出具体中医诊断建议
- **优化的模型加载机制**：单例模式模型加载和智能权重文件检测，提高系统响应速度

## 环境准备 🛠️

在运行系统前，请确保已安装以下依赖：

```bash
# 图像处理相关
pip install opencv-python
pip install scikit-image

# 机器学习相关
pip install tensorflow
pip install scikit-learn

# 数据处理相关
pip install numpy
pip install pandas
pip install openpyxl

# 可视化相关（可选）
pip install matplotlib
pip install seaborn
```

## 系统文件结构 📁

```
ChineseMedicine/
├── run.py                   # 主程序入口
├── ChineseMedicine_analysis.py   # 中医分析主模块
├── analysis/                # 分析模块目录
│   ├── main.py              # 健康值计算主函数
│   ├── merge_features.py    # 特征融合函数
│   └── LS.csv               # 经验权重数据
├── tongue/                  # 舌诊模块目录
│   ├── haveTongue.py        # 舌体检测模块
│   ├── segmentation_tongue.py   # 舌面分割模块
│   └── tongue_segmentation/ # 舌体区域分割模块
├── keras_segmentation/      # 图像分割模块
├── weights/                 # 模型权重目录
│   └── resunet.*.h5         # 模型权重文件（h5格式）
├── example/                 # 示例图片和结果目录
│   ├── *.jpg                # 测试用的舌像图片
│   └── expert_diagnosis.xlsx   # 诊断结果记录表
├── upload/                  # 上传图片临时存储目录（自动创建）
├── features/                # 特征存储目录（自动创建）
│   └── [用户ID]/            # 按用户ID分类的特征目录
│       └── value.txt        # 用户历史健康值记录
├── train/                   # 模型训练相关（自行创建）
│   ├── train.py             # 训练脚本
│   ├── plot_metrics.py      # 指标绘图脚本
│   └── configs/             # 训练配置文件
└── docs/                    # 文档目录（自行创建）
    ├── model_structure.md   # 模型结构文档
    ├── feature_extraction.md # 特征提取方法文档
    └── evaluation.md        # 评估方法文档
```

## 模型加载机制说明 🔄

系统采用了优化的模型加载机制，提高运行效率：

1. **单例模式**：使用全局变量确保模型只加载一次，避免重复加载
2. **高效权重查找**：采用直接文件名解析方式，快速定位最高轮次的权重文件
3. **模块间共享**：不同模块间共享同一个模型实例，减少内存占用
4. **懒加载机制**：只在实际需要使用时才加载模型，优化资源利用
5. **h5格式支持**：支持TensorFlow的h5格式权重文件，确保版本兼容性

此机制显著减少了系统启动和分析过程中的延迟。

## 模型测试与评估 🧪

除了主要的舌诊分析功能外，系统还提供了模型测试与评估工具：

### 使用test.py进行模型测试 🔬

您可以使用`test.py`脚本对训练好的模型进行性能测试：

```bash
python test.py
```

执行此命令会：

1. **加载最新的模型权重**: 自动查找并加载`weights`目录中轮次最高的h5格式权重文件
2. **预测测试集图像**: 对`tongue_data/test_img/`目录中的所有图像进行分割预测
3. **保存预测结果**: 将预测结果保存到`prediction/`目录
4. **评估模型性能**: 如果存在`tongue_data/test_label/`目录，会计算并输出模型的IoU指标

预测结果示例：
- 原始舌像图片：`tongue_data/test_img/101_org.jpg`
- 预测分割结果：`prediction/101_org.jpg`

> 📝 **注意**: 首次运行时，系统会自动检测并创建所需的目录结构

### 准备测试数据 📊

在使用测试功能前，请确保准备好测试数据：

1. 创建测试图像目录: `tongue_data/test_img/`
2. 创建测试标签目录: `tongue_data/test_label/` (用于评估，可选)
3. 将测试图像和对应标签放入相应目录

您也可以使用数据集划分工具自动生成这些目录:
```bash
python label/divide_datasets.py
```

### 解读测试结果 📈

测试完成后，您可以：

1. 检查 `prediction/` 目录查看分割结果
2. 查看输出的IoU值评估模型性能 (数值越接近1表示分割效果越好)
3. 比较预测结果与标签图像，分析模型的优势和不足

## 使用流程 🔄

### 1. 运行舌体检测与分析 👅

使用 `run.py` 文件进行舌体检测、分割和健康分析：

```bash
python run.py
```

#### 执行过程详解：

1. **输入图像** 📸
   - 系统默认使用 `example/0.jpg` 作为输入图像
   - 如需使用其他图像，请修改 `run.py` 中的 `image_path` 变量

2. **舌体检测** 🔍
   - 调用 `find_tongue()` 函数检测图像中的舌体
   - 系统会检查图像质量（亮度、对比度、是否含有舌体等）
   - 如果检测成功，会返回舌体区域的坐标和掩膜图像

3. **分割结果展示** 👁️
   - 系统会弹出窗口显示原图和舌体分割结果
   - 按任意键关闭窗口继续执行

4. **健康分析** 📊
   - 调用 `analysis()` 函数进行健康分析
   - 计算整体健康值和五脏（心、肝、脾、肺、肾）健康值
   - 数值范围为[-1, 1]，越接近0表示越健康

5. **诊断结果** 📝
   - 根据各脏腑健康值，系统会给出相应的中医诊断建议
   - 如"血虚"、"脾虚"、"肾虚"、"气虚"、"肝郁"等
   - 如果各项指标都正常，则诊断为"健康"

6. **结果保存** 💾
   - 分析结果会自动保存到 `example/expert_diagnosis.xlsx` 文件中
   - 每次分析都会新增一条记录，不会覆盖历史数据

### 2. 生成的文件及目录说明 📂

#### 自动创建的目录：

1. **upload/** 文件夹
   - 用于临时存储处理中的图像
   - 分析完成后会自动清理

2. **features/[用户ID]/** 文件夹
   - 存储用户健康值历史记录
   - 每个用户一个独立目录，以用户ID命名
   - 目录下的 `value.txt` 文件记录用户历次检测的健康值

#### 生成的文件：

1. **expert_diagnosis.xlsx**
   - 位置：`example/expert_diagnosis.xlsx`
   - 内容：所有用户的诊断记录
   - 字段包括：用户ID、检测时间、整体健康值、心、脾、肾、肺、肝、诊断结果

2. **value.txt**
   - 位置：`features/[用户ID]/value.txt`
   - 内容：特定用户的历史健康值记录
   - 格式：每行为一次检测结果，以逗号分隔的六个值（整体、心、脾、肾、肺、肝）

## 训练模型 🧠

本系统使用的舌体分割模型基于ResUNet架构，可以根据以下步骤训练自己的模型：

### 1. 数据集准备

首先使用数据集划分工具准备训练数据：

```bash
python label/divide_datasets.py
```

这会将数据集划分为训练集和测试集。确保您的数据集目录结构如下：

```
tongue_data/
├── train_img/      # 训练集图像
├── train_label/    # 训练集标签
├── test_img/       # 测试集图像
└── test_label/     # 测试集标签
```

### 2. 模型训练

使用以下命令开始训练：

```bash
python -m keras_segmentation.train \
    --checkpoints_path="weights/resunet" \
    --train_images="tongue_data/train_img/" \
    --train_annotations="tongue_data/train_label/" \
    --val_images="tongue_data/test_img/" \
    --val_annotations="tongue_data/test_label/" \
    --n_classes=2 \
    --input_height=512 \
    --input_width=512 \
    --model_name="resunet" \
    --batch_size=8 \
    --epochs=50
```

参数说明：
- `--checkpoints_path`: 模型权重保存路径
- `--train_images`: 训练集图像路径
- `--train_annotations`: 训练集标注路径
- `--val_images`: 验证集图像路径
- `--val_annotations`: 验证集标注路径
- `--n_classes`: 分类数量（2表示二分类：舌体和背景）
- `--input_height/width`: 输入图像尺寸
- `--model_name`: 模型架构名称
- `--batch_size`: 批次大小
- `--epochs`: 训练轮数

### 3. 模型评估

训练完成后，可以使用以下命令评估模型性能：

```bash
python -m keras_segmentation.predict \
    --checkpoints_path="weights/resunet" \
    --input_path="example/test.jpg" \
    --output_path="example/test_pred.png"
```

在实验报告中可以包含以下评估指标：
- IoU (Intersection over Union)
- F1-Score
- 精确率和召回率
- 分割可视化结果

### 4. 自定义训练脚本（可选）

可以创建自定义训练脚本 `train/train.py`，以实现更复杂的训练策略：

```python
# 示例代码片段
import tensorflow as tf
from keras_segmentation.models.resunet import resunet

# 数据增强
def data_augmentation(image, mask):
    # 随机翻转
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    
    # 随机亮度调整
    image = tf.image.random_brightness(image, 0.2)
    
    # 随机对比度调整
    image = tf.image.random_contrast(image, 0.8, 1.2)
    
    return image, mask

# 自定义损失函数
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# 创建并编译模型
model = resunet(n_classes=2, input_height=512, input_width=512)
model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])

# 训练模型
model.fit(train_generator, validation_data=val_generator, epochs=50, callbacks=[...])
```

## 高级功能和扩展 🚀

### 1. 批量处理和分析 📈

创建批处理脚本 `batch_process.py`，能够对整个文件夹中的舌像进行批量分析：

```python
import os
import glob
from run import analysis, find_tongue

def batch_process(folder_path, user_id):
    """批量处理文件夹中的所有图像"""
    results = []
    image_files = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                  glob.glob(os.path.join(folder_path, "*.png"))
    
    for image_path in image_files:
        print(f"处理图像: {os.path.basename(image_path)}")
        det_res = find_tongue(image_path)
        if len(det_res) != 2:
            print(f"舌体检测失败: {det_res}")
            continue
        
        result, mask = det_res
        analysis_result = analysis(image_path, user_id)
        results.append({
            "image": os.path.basename(image_path),
            "result": analysis_result
        })
    
    return results

# 使用示例
if __name__ == "__main__":
    results = batch_process("example/batch_images", "test_user")
    for res in results:
        print(f"图像: {res['image']}, 健康值: {res['result']['res']['healthy']}")
```

### 2. 时间序列分析 📅

添加一个时间序列分析模块，可以比较用户多次检测结果的变化趋势：

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_health_trend(user_id):
    """分析用户健康趋势"""
    excel_path = os.path.join("example", "expert_diagnosis.xlsx")
    if not os.path.exists(excel_path):
        return "无历史数据"
    
    df = pd.read_excel(excel_path)
    user_data = df[df['用户ID'] == user_id]
    
    if len(user_data) < 2:
        return "数据点不足，无法分析趋势"
    
    # 绘制时间序列图
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(user_data['检测时间'], user_data['整体健康值'], 'o-', label='整体健康值')
    plt.xlabel('检测时间')
    plt.ylabel('健康值')
    plt.title(f'用户 {user_id} 的健康趋势')
    plt.grid(True)
    plt.legend()
    
    # 绘制各器官健康值热力图
    plt.subplot(2, 1, 2)
    organ_data = user_data[['心', '脾', '肾', '肺', '肝']]
    sns.heatmap(organ_data.T, annot=True, cmap='RdYlGn', 
                xticklabels=user_data['检测时间'], yticklabels=['心', '脾', '肾', '肺', '肝'])
    plt.title('五脏健康值变化')
    plt.tight_layout()
    
    plt.savefig(f"example/trend_{user_id}.png")
    return f"趋势分析已保存至 example/trend_{user_id}.png"
```

### 3. 舌象特征可视化 🌈

添加一个模块，将舌体的特征进行可视化展示：

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_tongue_features(image_path, mask):
    """舌象特征可视化"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 提取舌体区域
    tongue_region = cv2.bitwise_and(img, img, mask=mask)
    
    # 计算舌体颜色直方图
    hsv_tongue = cv2.cvtColor(tongue_region, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv_tongue], [0], mask, [180], [0, 180])
    s_hist = cv2.calcHist([hsv_tongue], [1], mask, [256], [0, 256])
    v_hist = cv2.calcHist([hsv_tongue], [2], mask, [256], [0, 256])
    
    # 绘制可视化图
    plt.figure(figsize=(15, 10))
    
    # 原图和分割结果
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(tongue_region)
    plt.title('舌体区域')
    plt.axis('off')
    
    # 颜色分布直方图
    plt.subplot(2, 3, 3)
    plt.plot(h_hist, color='r', label='H通道')
    plt.plot(s_hist, color='g', label='S通道')
    plt.plot(v_hist, color='b', label='V通道')
    plt.title('颜色分布直方图')
    plt.legend()
    
    # 舌体表面纹理分析
    plt.subplot(2, 3, 4)
    gray_tongue = cv2.cvtColor(tongue_region, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_tongue, 50, 150)
    plt.imshow(edges, cmap='gray')
    plt.title('舌体边缘特征')
    plt.axis('off')
    
    # 保存结果
    plt.tight_layout()
    plt.savefig(f"example/features_{os.path.basename(image_path)}.png")
    
    return f"特征可视化已保存至 example/features_{os.path.basename(image_path)}.png"
```

## 实验分析与评估 📝

### 1. 模型性能评估

对舌体分割模型进行性能评估是实验报告的重要部分。建议包含以下内容：

- **量化指标**：计算并报告IoU、精确率、召回率、F1值等指标
- **可视化比较**：对比原始图像、真实标签和模型预测结果
- **消融实验**：尝试不同的模型架构或参数设置，比较性能差异
- **混淆矩阵**：分析分类错误的模式和原因

### 2. 临床有效性评估

与传统中医舌诊结果进行对比，评估系统的临床有效性：

- **专家对比**：将系统诊断结果与中医专家诊断结果进行对比
- **一致性分析**：计算系统与专家诊断的一致性指标（如Kappa系数）
- **案例分析**：选择典型案例详细分析系统诊断结果与专家意见的异同
- **局限性讨论**：分析当前系统的局限性和可能的改进方向

### 3. 系统技术亮点

在实验报告中重点强调以下技术亮点：

- **多任务学习**：同时进行舌体分割和健康状态评估
- **领域知识融合**：将传统中医理论与现代机器学习技术结合
- **个性化分析**：基于用户历史数据进行个性化健康评估
- **可解释性**：系统不仅给出健康评分，还提供具体的诊断建议
- **端到端方案**：从图像输入到健康评估的完整解决方案

## 数据集划分工具 🔀

系统还提供了数据集划分工具 `label/divide_datasets.py`，用于将舌像数据集分为训练集和测试集：

```bash
python label/divide_datasets.py
```

### 执行结果：

- 将 `tongue_images/` 目录下的图像和 `tongue_labels/` 目录下的标签
- 按 9:1 的比例随机划分为训练集和测试集
- 训练集存储在 `tongue_data/train_img/` 和 `tongue_data/train_label/`
- 测试集存储在 `tongue_data/test_img/` 和 `tongue_data/test_label/`

## 注意事项 ⚠️

1. 舌像图片需要光线适中，清晰可见，且舌体占据图像主要部分
2. 首次使用时会自动下载和配置模型权重
3. 诊断结果仅供参考，不能替代专业医生的诊断
4. 使用前请备份重要数据，尤其是运行数据集划分工具时
5. 模型训练需要较高的计算资源，建议使用GPU加速训练过程

## 常见问题解答 ❓

1. **图像检测失败怎么办？**
   - 请确保图像清晰、光线适中，且舌体占据图像主要部分
   - 尝试调整图像的亮度、对比度或更换图像

2. **如何为不同用户进行分析？**
   - 修改 `run.py` 中 `analysis()` 函数调用时的 `user_id` 参数

3. **如何查看历史分析结果？**
   - 打开 `example/expert_diagnosis.xlsx` 文件查看所有分析记录

4. **如何重新训练模型？**
   - 按照"训练模型"章节的步骤进行训练

5. **如何调整诊断阈值？**
   - 在 `run.py` 中修改相应的阈值参数，如 `abs(result2['res']['heart']) > 0.4` 中的 0.4

## 未来工作展望 🔮

1. **多模态融合**：结合舌诊、脉诊等多种中医诊断方法
2. **移动端部署**：开发移动应用，方便用户自主检测
3. **多语言支持**：支持中英等多种语言界面
4. **个性化推荐**：基于诊断结果推荐个性化的保健建议
5. **数据挖掘**：对大量用户数据进行挖掘，发现疾病模式

## 参考文献 📚

1. 王忠民. 舌诊在中医诊断中的应用研究[J]. 中医学报, 2018, 33(5): 825-828.
2. Zhang B, Kumar B V, Zhang D. Detecting diabetes mellitus and nonproliferative diabetic retinopathy using tongue color, texture, and geometry features[J]. IEEE Transactions on Biomedical Engineering, 2013, 61(2): 491-501.
3. Ronneberger O, Fischer P, Brox T. U-net: Convolutional networks for biomedical image segmentation[C]//International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015: 234-241.
4. 李松, 张伟, 王彤. 基于深度学习的中医舌象客观化研究进展[J]. 中国中医药信息杂志, 2020, 27(3): 113-117. 