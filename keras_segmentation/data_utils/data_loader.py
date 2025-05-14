import itertools
import os
import random
import six
import numpy as np
import cv2
from tqdm import tqdm

DATA_LOADER_SEED = 0 
random.seed(DATA_LOADER_SEED)  # 设置随机种子，使得随机数据可预测，即只要seed的值一样，后续生成的随机数都一样
class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(10)]

class DataLoaderError(Exception): # 抛出异常
    pass


# 在检查数据完整性的同时，查找目录下的所有舌体图像文件，以及对应的标签文件
def get_pairs_from_paths(images_path, segs_path, ignore_non_matching=False):

    ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]   # 接受图像的格式
    ACCEPTABLE_SEGMENTATION_FORMATS = [".png", ".bmp", '.jpg']       # 接受标签图像的格式

    image_files = [] # 定义空列表，用于存储舌体图像的路径信息
    segmentation_files = {} # 用于存放标签图像的路径信息

    for dir_entry in os.listdir(images_path):   # 遍历所有的图像
        if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS: # 如果图像的格式符合要求
            file_name, file_extension = os.path.splitext(dir_entry) # 使用os.path.splitext()方法，获取图像的名称和扩展名
            image_files.append((file_name, file_extension,
                                os.path.join(images_path, dir_entry))) # 将图像的文件名、扩展名、图像的路径作为一个对象保存在image_files列表中

    for dir_entry in os.listdir(segs_path): # 遍历所有的标签文件
        if os.path.isfile(os.path.join(segs_path, dir_entry)) and \
           os.path.splitext(dir_entry)[1] in ACCEPTABLE_SEGMENTATION_FORMATS: # 如果标签文件的格式符合要求
            file_name, file_extension = os.path.splitext(dir_entry) # 使用os.path.splitext()方法，获取标签图像的名称和扩展名
            full_dir_entry = os.path.join(segs_path, dir_entry) # 标签图像的路径
            # print(file_name)
            if file_name in segmentation_files: # 如果在标签文件中已经存在该标签图像，则调用DataLoaderError类抛出异常，并附带异常的描述信息
                raise DataLoaderError("Segmentation file with filename {0}"
                                      " already exists and is ambiguous to"
                                      " resolve with path {1}."
                                      " Please remove or rename the latter."
                                      .format(file_name, full_dir_entry))


            segmentation_files[file_name] = (file_extension, full_dir_entry)# 如果标签文件名不在标签文件中，就将该标签文件的文件扩展名及文件路径保存在标签文件中

    return_value = [] # 初始化空列表，用于存放图像的路径信息及其标签的路径信息
    # 匹配图像及其标签文件
    for image_file, _, image_full_path in image_files: # 遍历image_files列表
        if image_file in segmentation_files: # 如果当前图像文件的名称在标签文件中
            return_value.append((image_full_path,
                                segmentation_files[image_file][1])) # 则将图像的路径信息和标签图像的路径信息添加到return_value列表中
        elif ignore_non_matching: # 如果已经匹配过
            continue # 跳出本次循环
        else: # 如果不匹配，则抛出异常
            raise DataLoaderError("No corresponding segmentation "
                                  "found for image {0}."
                                  .format(image_full_path))

    return return_value # 返回图像的路径信息及其标签的路径信息

# 标准化处理图像
def get_image_array(image_input, width, height, imgNorm = "sub_mean"):  
    if type(image_input) is np.ndarray:  # 如果读取的图像数据是多维数组
        img = image_input  # 将图像数据保存在img变量中
    elif isinstance(image_input, six.string_types):  # 判断传入的image_input参数是不是str类型
        if not os.path.isfile(image_input):  # 如果不存在该路径，抛出异常
            raise DataLoaderError("get_image_array: path {0} doesn't exist"
                                  .format(image_input))
            # 抛出DataLoaderError关于path {0} doesn't exist的异常
        img = cv2.imread(image_input, 1)  # 存在，则使用cv2.imread()方法读取图像
    else:# 若上述条件都不满足，则抛出无法处理该输入图像的异常
        raise DataLoaderError("get_image_array: Can't process input type {0}"
                              .format(str(type(image_input))))
    if imgNorm == "sub_mean":   # 如果传入的参数为sub_mean
        img = cv2.resize(img, (width, height))  # 将图像调整到标准尺寸
        img = img.astype(np.float32)  # 将img的数据类型转换为float32
        img[:, :, 0] -= 103.939  # 图像的第0(R)通道
        img[:, :, 1] -= 116.779  # 图像的第1(G)通道
        img[:, :, 2] -= 123.68  # 图像的第2(B)通道
        img = img[:, :, ::-1]  # 转换图像通道，通道顺序由0，1，2转换为2，1，0

    return img

# 标准化处理标签图像
def get_segmentation_array(image_input, nClasses,
                           width, height, no_reshape=False):
    """ Load segmentation array from input """

    seg_labels = np.zeros((height, width, nClasses)) # 创建一个nClasses通道的height* width大小的全零数组

    if type(image_input) is np.ndarray: # 如果读取的图像数据是多维数组
        img = image_input # 将图像数据保存在img变量中
    elif isinstance(image_input, six.string_types): # 判断传入的image_input参数是不是str类型
        if not os.path.isfile(image_input): # 如果不存在该路径，抛出异常
            raise DataLoaderError("get_segmentation_array: "
                                  "path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 1) # 存在，则使用cv2.imread()方法读取标签图像
    else: # 若上述条件都不满足，则抛出无法处理该输入图像的异常
        raise DataLoaderError("get_segmentation_array: "
                              "Can't process input type {0}"
                              .format(str(type(image_input))))

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST) # 将标签图像调整到标准尺寸，使用最邻近插值的方法
    img = img[:, :, 0] # 获取图像的第0通道
    img[img>1] = 1 # 如果像素值大于1，则赋值为1
    for c in range(nClasses): # 为创建的全零数组赋值
        seg_labels[:, :, c] = (img == c).astype(int) #将img的值转换为int类型，并赋值给对应的通道

    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width*height, nClasses)) # 使用reshape()方法将多维数组指定的形状

    return seg_labels

# 数据增强
def image_segmentation_generator(images_path, segs_path, batch_size,
                                 n_classes, input_height, input_width,
                                 output_height, output_width
                                 ):

    img_seg_pairs = get_pairs_from_paths(images_path, segs_path) # 调用定义的get_pairs_from_paths()方法，在检查数据完整性的同时，查找目录下的所有舌体图像文件，以及对应的标签文件
    random.shuffle(img_seg_pairs) # 随机打乱数据的顺序
    zipped = itertools.cycle(img_seg_pairs) # 使用itertools.cycle()方法，创建一个迭代对象，对于输入的iterable的元素反复执行循环操作

    while True:
        X = [] # 定义一个空列表，用于存放图像数据
        Y = [] # 定义一个空列表，用于存放标签数据
        for _ in range(batch_size): # 遍历每个batch_size中的图像
            im, seg = next(zipped) # 使用python3的内置方法next()，返回迭代器的下一个项目

            im = cv2.imread(im, 1) # 读取图像
            seg = cv2.imread(seg, 1) # 读取标签图像

            X.append(get_image_array(im, input_width,
                                     input_height)) # 将读取的图像数据进行标准化处理后，存入X列表中
            Y.append(get_segmentation_array(
                seg, n_classes, output_width, output_height)) # 将读取的标签数据进行标准化处理后，存入Y列表中

        yield np.array(X), np.array(Y)