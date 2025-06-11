import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings

warnings.filterwarnings("ignore")

from keras_segmentation.models.unet import resnet50_unet
# -*- coding: utf-8 -*-
from keras_segmentation.models.unet import resnet50_unet

# 调用定义好的resnet50_unet网络结构，并传入参数
model = resnet50_unet(n_classes=2, input_height=576, input_width=768)
# 训练res_unet模型
model.train(
    train_images="tongue_data/train_img/",  # 训练图像的路径
    train_annotations="tongue_data/train_label/",  # 训练图像标签的路径
    checkpoints_path="weights/resunet",  # 训练的模型保存路径
    epochs=50,  # 训练的轮次
    batch_size=2,  # 每个batch读入2张图片
    steps_per_epoch=340,  # 在训练集上每个epoch的迭代次数
    val_steps_per_epoch=40,  # 在验证集上每个epoch的迭代次数
    val_images='tongue_data/test_img',  # 验证图像的路径
    val_annotations='tongue_data/test_label',  # 验证图像标签的路径
    validate=True  # 是否有验证集
)
