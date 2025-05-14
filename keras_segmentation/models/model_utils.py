from types import MethodType
from keras.models import *
from keras.layers import *
from ..train import train
from ..predict import predict

def get_segmentation_model(input, output):
   img_input = input  # 输入数据
   o = output # 输出数据
   o_shape = Model(img_input, o).output_shape  # 定义模型的输出数据
   i_shape = Model(img_input, o).input_shape  # 定义模型的输入数据
   output_height = o_shape[1]  # 将模型输出数据的第一个作为输出数据的高
   output_width = o_shape[2]  # 将模型输出数据的第二个作为输出数据的宽
   input_height = i_shape[1]  # 将模型输入数据的第一个作为输入数据的高
   input_width = i_shape[2]  # 将模型输入数据的第二个作为输入数据的宽
   n_classes = o_shape[3]  # 将模型输出数据的第三个作为类别数
   o = (Reshape((output_height*output_width, -1)))(o)  # 将输出数据改成一串，没有行列
   o = (Activation('softmax'))(o)
   # 将多个神经元的输出，映射到（0,1）区间内，可以看成概率来理解，进行多分类
   model = Model(img_input, o)    # 定义模型
   model.output_width = output_width  # 定义模型输出数据的宽
   model.output_height = output_height  # 定义模型输出数据的宽
   model.n_classes = n_classes  # 定义模型的类别数
   model.input_height = input_height  # 定义模型输入数据的高
   model.input_width = input_width  # 定义模型输入数据的宽
   model.model_name = ""  # 定义模型名
   model.train = MethodType(train, model)  # 模型训练
   model.predict_segmentation = MethodType(predict, model)  # 当前图像预测结果
   return model