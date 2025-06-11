from keras_segmentation.predict import evaluate
import os
from keras_segmentation.models.unet import resnet50_unet

# 导入我们优化过的模型加载机制
from ChineseMedicine_analysis import get_model


def test_img(model, input_path, output_path):  # 用训练好的模型对图像进行测试
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)

    input_list = os.listdir(input_path)  # 获取测试图像的路径信息
    for i in input_list:
        print(f"处理图像: {i}")
        # 使用predict_segmentation方法进行预测
        model.predict_segmentation(  # 输出当前图像预测结果
            inp=os.path.join(input_path, i),  # 输入图像路径
            out_fname=os.path.join(output_path, i)  # 输出图像路径
        )


if __name__ == '__main__':
    print("正在加载模型...")
    # 使用优化后的模型加载函数获取模型实例
    model = get_model()

    if model is None:
        print("模型加载失败，请确保权重文件存在")
        exit(1)

    # 检查测试目录
    test_img_dir = 'tongue_data/test_img/'
    test_label_dir = 'tongue_data/test_label/'
    prediction_dir = 'prediction/'

    # 检查测试目录是否存在
    if not os.path.exists(test_img_dir):
        print(f"测试图像目录 {test_img_dir} 不存在，请先准备测试数据")
        exit(1)

    # 运行测试
    print(f"\n开始预测测试集图像...")
    test_img(model, test_img_dir, prediction_dir)

    # 如果测试标签目录存在，则评估模型性能
    if os.path.exists(test_label_dir):
        print(f"\n开始评估模型性能...")
        res = evaluate(
            model=model,
            inp_images_dir=test_img_dir,
            annotations_dir=test_label_dir
        )
        print("\n模型评估结果:")
        print(f"IoU: {res}")
    else:
        print(f"\n测试标签目录 {test_label_dir} 不存在，跳过模型评估")

    print("\n测试完成!")
