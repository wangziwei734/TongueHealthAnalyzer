from PIL import Image
import matplotlib.pyplot as plt
import os

img_path = 'tongue_data/test_img/'   # 测试数据存放路径
mask_path = 'prediction/'  # 预测数据存放路径
out_path = 'compare/'  # 测试图像与预测图象组合的对比图像保存路径


img_list = os.listdir(img_path)  # 获取测试图像目录下的所有文件
for i in img_list: # 遍历测试图像
    img = Image.open(img_path + i)  # 读取测试图像
    mask = Image.open(mask_path + i)  # 读取预测图像
    print(out_path + i.split('.')[0] + '.jpg')

    plt.subplot(1, 2, 1)  # 该图有1行2列，此图是第1个图。
    plt.imshow(img)      # 显示图片
    plt.title('img', fontsize=8)   # 声明原图标题为img 字体大小为8 `
    plt.xticks([])    # 给原图创建x轴与y轴标签
    plt.yticks([])

    plt.subplot(1, 2, 2)  # 该图有1行2列，此图是第2个图。
    plt.imshow(mask)    # 显示图片
    plt.title('pre', fontsize=8)  # 声明预测图片标题为pre 字体大小为8
    plt.xticks([])    # 给预测图片创建x轴与y轴标签
    plt.yticks([])

    plt.savefig(out_path + i.split('.')[0] + '.jpg')   # 保存合并后的图像到compare目录下
    
    
    