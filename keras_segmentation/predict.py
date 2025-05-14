import random
import json
import os
import six
import cv2
import numpy as np
from .train import find_latest_checkpoint
from tqdm import tqdm
from .data_utils.data_loader import get_image_array,get_segmentation_array,DATA_LOADER_SEED, class_colors,get_pairs_from_paths
random.seed(DATA_LOADER_SEED)

def model_from_checkpoint_path(checkpoints_path):
    from .models.all_models import model_from_name
    assert (os.path.isfile(checkpoints_path+"_config.json")), "Checkpoint not found."
    model_config = json.loads(open(checkpoints_path+"_config.json", "r").read())
    latest_weights = find_latest_checkpoint(checkpoints_path)
    assert (latest_weights is not None), "Checkpoint not found."
    
    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], 
        input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    
    print("尝试加载权重文件:", latest_weights)
    
    # 多种方式尝试加载权重
    load_status = False
    error_msgs = []
    
    # 方式1: 直接加载
    try:
        model.load_weights(latest_weights)
        print("✓ 使用默认方式加载成功")
        load_status = True
    except Exception as e:
        error_msgs.append(f"默认加载失败: {str(e)}")
        
        # 方式2: 使用h5py直接读取
        try:
            import h5py
            with h5py.File(latest_weights, 'r') as f:
                weight_names = [n.decode('utf8') for n in f.attrs['weight_names']]
                for name in weight_names:
                    layer_weights = [f[name][i] for i in range(len(f[name]))]
                    model.get_layer(name.split('/')[0]).set_weights(layer_weights)
            print("✓ 使用h5py方式加载成功")
            load_status = True
        except Exception as e:
            error_msgs.append(f"h5py加载失败: {str(e)}")
            
            # 方式3: 尝试转换后加载
            try:
                model.load_weights(latest_weights, by_name=True)
                print("✓ 使用by_name方式加载成功")
                load_status = True
            except Exception as e:
                error_msgs.append(f"by_name加载失败: {str(e)}")
    
    if not load_status:
        print("❌ 所有加载方式均失败:")
        for msg in error_msgs:
            print(f"  - {msg}")
        raise Exception("无法加载权重文件")
        
    return model

def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]  # 输出图像的高
    output_width = seg_arr.shape[1]  # 输出图像的宽

    seg_img = np.zeros((output_height, output_width, 3))  # 创建一个3通道的output_height行，output_width列的数组

    for c in range(n_classes):  
        seg_arr_c = seg_arr[:, :] == c  
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')  # 计算通道0的数据
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')  # 计算通道1的数据
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')  # 计算通道2 的数据

    return seg_img

# 可视化分割结果
def visualize_segmentation(seg_arr, inp_img=None, n_classes=None,
                           colors=class_colors):

    if n_classes is None:
        n_classes = np.max(seg_arr)  # 调用np.max()方法返回numpy数组中的最大值

    seg_img = get_colored_segmentation_image(seg_arr, n_classes, colors=colors)# 调用定义的get_colored_segmentation_image()函数，为分割结果创建3通道的数组，并赋值颜色数据

    if inp_img is not None:
        orininal_h = inp_img.shape[0]  # 输入图像的宽
        orininal_w = inp_img.shape[1]  # 输入图像的高
        seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))  # 将分割结果图像resize为输入图像的大小
    return seg_img


# 调用训练好的模型，对图像进行预测
def predict(model=None, inp=None, out_fname=None):
    assert (inp is not None)  # 判断输入图像
    assert ((type(inp) is np.ndarray) or isinstance(inp, six.string_types)),\
        "Input should be the CV image or the input file name"
    
    if isinstance(inp, six.string_types):  # 判断输入图像的类型是否为字符串类型
        inp = cv2.imread(inp)  # 使用cv2.imread()方法读取图像
    
    # 判断输入的图像是不是包含h,w，3三个信息
    assert len(inp.shape) == 3, "Image should be h,w,3 " 
    

    output_width = model.output_width  # 输出图像的宽
    output_height = model.output_height  # 输出图像的高
    input_width = model.input_width  # 输入图像的宽
    input_height = model.input_height  # 输入图像的高
    n_classes = model.n_classes  # 类别数

    x = get_image_array(inp, input_width, input_height,)  # 调用定义的get_image_array()方法，对输入图像进行标准化处理
    pr = model.predict(np.array([x]))[0]  # 模型预测
    pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2) # 将pr的维度变为output_height行output_width列且求参数(集合)
    seg_img = visualize_segmentation(pr, inp, n_classes=n_classes,
                                     colors=[(0, 0, 0), (255, 255, 255)])# 调用定义的visualize_segmentation()方法，可视化舌体分割图像

    if out_fname is not None:  # 如果输出图像存在
        cv2.imwrite(out_fname, seg_img)  # 则将可视化的分割图像保存

    return pr, seg_img


# 定义评价指标miou，用于评估模型的分割性能
def evaluate(model=None, inp_images=None, annotations=None,
             inp_images_dir=None, annotations_dir=None, checkpoints_path=None):
    if model is None: 
        assert (checkpoints_path is not None),\
                "Please provide the model or the checkpoints_path"
        model = model_from_checkpoint_path(checkpoints_path) # 调用定义的model_from_checkpoint_path()函数，寻找训练轮次最多的模型

    if inp_images is None: 
        assert (inp_images_dir is not None),\
                "Please provide inp_images or inp_images_dir"
        assert (annotations_dir is not None),\
            "Please provide inp_images or inp_images_dir"
        
        paths = get_pairs_from_paths(inp_images_dir, annotations_dir) # 调用定义好的get_pairs_from_paths()方法，在检查数据完整性的同时，查找测试图像及其标签图像
        paths = list(zip(*paths)) # 将测试图像及标签图像的路径存入列表中
        inp_images = list(paths[0]) # 将测试图像的路径信息保存到inp_images变量中
        annotations = list(paths[1]) # 将标签图像的路径信息保存到annotations变量中

    assert type(inp_images) is list # 判断inp_images变量是否是列表类型
    assert type(annotations) is list # 判断annotations变量是否是列表类型

    tp = np.zeros(model.n_classes) 
    fp = np.zeros(model.n_classes) 
    fn = np.zeros(model.n_classes)

    for inp, ann in tqdm(zip(inp_images, annotations)):
        pr,_ = predict(model, inp) # 预测
        gt = get_segmentation_array(ann, model.n_classes,
                                    model.output_width, model.output_height,
                                    no_reshape=True) # 调用定义好的get_segmentation_array()函数，读取标签图像，并进行标准化处理
        gt = gt.argmax(-1) # 
        pr = pr.flatten() # 使用flatten()方法，将预测结果展平
        gt = gt.flatten() # 使用flatten()方法，将标签展平

        for cl_i in range(model.n_classes):

            tp[cl_i] += np.sum((pr == cl_i) * (gt == cl_i)) # 计算交集的和
            fp[cl_i] += np.sum((pr == cl_i) * ((gt != cl_i))) # 计算（pr与gt的差集-交集）的和
            fn[cl_i] += np.sum((pr != cl_i) * ((gt == cl_i))) # 计算（gt与pr的差集-交集）和

    cl_wise_score = tp / (tp + fp + fn + 0.000000000001) # 计算交并比，其中0.000000000001的作用是防止分母为0
    IOU_1 = cl_wise_score[1]

    return {"IOU": IOU_1}