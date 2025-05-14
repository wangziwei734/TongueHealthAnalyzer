from keras.layers import *
from .model_utils import get_segmentation_model
from .resnet50 import get_resnet50_encoder

IMAGE_ORDERING = 'channels_last'
MERGE_AXIS = -1

def _unet(n_classes, encoder, l1_skip_conn=True, input_height=416,input_width=608):
    # 输入类别数，编码器，输入及输出尺寸
    img_input, levels = encoder(
        input_height=input_height, input_width=input_width) 
    # 从encoder获取模型输入层和卷积块
    [f1, f2, f3, f4, f5] = levels
    o = f4
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o) # 卷积前进行Zeropadding填充
    o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o) # 添加二维卷积层，512个3x3的卷积核
    o = (BatchNormalization())(o) # 添加BN层

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o) # 上采样
    o = (concatenate([o, f3], axis=MERGE_AXIS)) # 对两个输出进行合并，做特征融合
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o) # 卷积前进行Zeropadding填充
    o = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o) # 添加二维卷积层，256个3x3的卷积核
    o = (BatchNormalization())(o) # 添加BN层

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o) # 上采样
    o = (concatenate([o, f2], axis=MERGE_AXIS)) # 对两个输出进行合并
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o) # 卷积前进行Zeropadding填充
    o = (Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)# 添加二维卷积层，128个3x3的卷积核
    o = (BatchNormalization())(o) # 添加BN层

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o) # 上采样

    if l1_skip_conn:
        o = (concatenate([o, f1], axis=MERGE_AXIS)) # 对两个输出进行合并

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o) # 卷积前进行Zeropadding填充
    o = (Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o) # 添加二维卷积层，64个3x3的卷积核
    o = (BatchNormalization())(o) # 添加BN层

    o = Conv2D(n_classes, (3, 3), padding='same',data_format=IMAGE_ORDERING)(o) # 添加二维卷积层，2个3x3的卷积核
    model = get_segmentation_model(img_input, o)    # 指定输入输出，得到模型
    return model

def resnet50_unet(n_classes, input_height=416, input_width=608,
                  encoder_level=3):
    model = _unet(n_classes, get_resnet50_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "resnet50_unet"
    return model