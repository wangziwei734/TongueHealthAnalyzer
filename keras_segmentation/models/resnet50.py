import keras
from keras.models import *
from keras.layers import *
from keras import layers

IMAGE_ORDERING = 'channels_last'
pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                 "releases/download/v0.2/" \
                 "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5" # 无全连接层的resnet50的预训练权重的下载地址

def one_side_pad(x):
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x) # 使用ZeroPadding2D()方法为特征图的上下左右各补充一行一列0，特征图的行数+2，列数+2
    x = Lambda(lambda x: x[:, :-1, :-1, :])(x)  # 将右侧和下侧补充的0去除
    return x


# 定义identity_block，identity block是维度一致的res block，最后将输出与input_tensor直接相加
def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters # 获取卷积核的数量
    bn_axis = 3
    # 定义卷积层和BN层的名称
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # 主干的第一部分：Conv2D（1x1）+BN+RELU
    x = Conv2D(filters1, (1, 1), data_format=IMAGE_ORDERING,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    # 主干的第二部分：Conv2D(3x3)+BN+RELU
    x = Conv2D(filters2, kernel_size, data_format=IMAGE_ORDERING,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    # 主干的第三部分：Conv2D（1x1）+BN
    x = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING,
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    # Add(x+input_tensor)+RELU
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x   

def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2)):
    # Conv Block残差块
    filters1, filters2, filters3 = filters  # 过滤器
    bn_axis = 3
    # 定义基本的名字
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # 网络主干的第一部分
    x = Conv2D(filters1, (1, 1), data_format=IMAGE_ORDERING, strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    # 网络主干的第二部分
    x = Conv2D(filters2, kernel_size, data_format=IMAGE_ORDERING,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    # 网络主干的第三部分
    x = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING,
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING,
                      strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)
    #  网络主干的最后部分,为网络主干添加shortcut并通过relu激活
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def get_resnet50_encoder(input_height=224, input_width=224, pretrained='imagenet'):
    # 编码器实现
    assert input_height % 32 == 0  # 检验输入图像的高对32求余是否为0
    assert input_width % 32 == 0  # 检验输入图像的宽对32求余是否为0
    img_input = Input(shape=(input_height, input_width, 3))
    # 接受检验过的输入图像的高宽，返回为 string 类型
    bn_axis = 3 # 三维

    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(img_input)  # 零填充
    # resnet50网络结构中的Stage 0，对输入图像数据的预处理
    x = Conv2D(64, (7, 7), data_format=IMAGE_ORDERING,
               strides=(2, 2), name='conv1')(x)
    f1 = x
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), data_format=IMAGE_ORDERING, strides=(2, 2))(x)
    # resnet50网络结构中的Stage 1
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    f2 = one_side_pad(x)
    # resnet50网络结构中的Stage 2
    # Conv Block残差块使用三组大小为 [128，128，512] 的滤波器
    # 3个Identity Block 残差块使用三组大小为 [128，128，512] 的筛选器
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    f3 = x
    # resnet50网络结构中的Stage 3 (≈6 lines)
    # Conv Block残差块使用三组大小为 [256， 256， 1024] 的滤波器。
    # 5 个Identity Block 残差块使用三组大小为 [256， 256， 1024] 的筛选器。
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    f4 = x
    # resnet50网络结构中的Stage 4
    # Conv Block残差块使用三组大小为 [512， 512， 2048] 的滤波器。
    # 2 个Identity Block 残差块使用三组大小为 [256， 256， 2048] 的过滤器。
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    f5 = x
    # 过滤器为[256，256，2048]，但无法分级。使用[512，512，2048]通过评分
    # 池化
    # 2D平均池化使用窗口的形状为（2，2），其名称为“avg_pool”。
    x = AveragePooling2D(
        (7, 7), data_format=IMAGE_ORDERING, name='avg_pool')(x)
    # f6 = x
    if pretrained == 'imagenet':
        weights_path = keras.utils.get_file(
           pretrained_url.split("/")[-1], pretrained_url)
        Model(img_input, x).load_weights(weights_path)

    return img_input, [f1, f2, f3, f4, f5]