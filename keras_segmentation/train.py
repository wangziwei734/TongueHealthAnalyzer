import json
from .data_utils.data_loader import image_segmentation_generator
import glob
from keras.callbacks import Callback
from keras.losses import categorical_crossentropy


def find_latest_checkpoint(checkpoints_path, fail_safe=True):
    def get_epoch_number_from_path(path):  # 定义get_epoch_number_from_path()函数，用于分割模型的路径，获得路径的单词列表
        return path.rsplit('.', 1)[-1]  # 使用rsplit()方法，以'.'作为分割符，得到字符串中的单词列表

    all_checkpoint_files = glob.glob(checkpoints_path + ".*")  # 获取所有匹配的文件列表
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f)
                                       .isdigit(), all_checkpoint_files))  # 过滤掉epoc_number部分为纯数字的条目
    if not len(all_checkpoint_files):  # 如果没有匹配到任何文件，则输出以下信息
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid"
                             .format(checkpoints_path))
        else:
            return None

    # 找到训练轮次最多的文件
    latest_epoch_checkpoint = max(all_checkpoint_files,
                                  key=lambda f:
                                  int(get_epoch_number_from_path(f)))
    return latest_epoch_checkpoint


# 定义交叉熵损失函数
def masked_categorical_crossentropy(gt, pr):
    mask = 1 - gt[:, :, 0]
    return categorical_crossentropy(gt, pr) * mask


# 保存训练过程中的模型
class CheckpointsCallback(Callback):
    def __init__(self, checkpoints_path):
        self.checkpoints_path = checkpoints_path

    def on_epoch_end(self, epoch, logs=None):
        if self.checkpoints_path is not None:
            # 使用h5格式保存
            self.model.save_weights(self.checkpoints_path + "." + str(epoch) + ".h5", save_format='h5')
            print("saved ", self.checkpoints_path + "." + str(epoch) + ".h5")


def train(model,  # 模型
          train_images,  # 训练集
          train_annotations,  # 训练集标签
          checkpoints_path=None,  # 模型保存的路径
          epochs=5,  # 训练伦次
          batch_size=2,  # 每次加载多少张图象
          validate=False,  # 是否进行验证
          val_images=None,  # 验证集
          val_annotations=None,  # 验证集标签
          val_batch_size=2,  # 每次加载多少张验证图像
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=480,  # 每个epoch在训练集上迭代多少次
          val_steps_per_epoch=60,  # 每个epoch在验证集上迭代多少次
          gen_use_multiprocessing=False,
          ignore_zero_class=False,
          optimizer_name='adadelta'  # 优化器
          ):
    n_classes = model.n_classes  # 数据集中的类别数
    input_height = model.input_height  # 输入图像的高
    input_width = model.input_width  # 输入图像的宽
    output_height = model.output_height  # 输出图像的高
    output_width = model.output_width  # 输出图像的宽

    if validate:
        assert val_images is not None
        assert val_annotations is not None

    if optimizer_name is not None:  # 如果优化器不为空

        if ignore_zero_class:
            loss_k = masked_categorical_crossentropy  # 调用定义的masked_categorical_crossentropy()函数，计算loss
        else:
            loss_k = 'categorical_crossentropy'  # 使用categorical_crossentropy()函数，计算loss

        # 编译模型
        model.compile(loss=loss_k,
                      optimizer=optimizer_name,
                      metrics=['accuracy'])

    if checkpoints_path is not None:  # 如果存在保存模型的路径
        with open(checkpoints_path + "_config.json", "w") as f:  # 打开模型保存目录下面的json文件进行写操作
            json.dump({
                "model_class": model.model_name,  # 模型名称
                "n_classes": n_classes,  # 类别数量
                "input_height": input_height,  # 输入图像的高
                "input_width": input_width,  # 输入图像的宽
                "output_height": output_height,  # 输出图像的高
                "output_width": output_width  # 输出图像的宽
            }, f)

    if load_weights is not None and len(load_weights) > 0:  # 如果存在预训练的权重
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)  # 加载模型

    if auto_resume_checkpoint and (checkpoints_path is not None):
        latest_checkpoint = find_latest_checkpoint(
            checkpoints_path)  # 调用定义的find_latest_checkpoint()方法，寻找训练轮次最多的模型，即最后近一次训练的模型
        if latest_checkpoint is not None:  # 如果找到了最近一次训练的模型
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)  # 加载该模型

    # 对训练集进行数据增强
    train_gen = image_segmentation_generator(
        train_images, train_annotations, batch_size, n_classes,
        input_height, input_width, output_height, output_width)

    if validate:  # 如果验证集存在，则对验证集进行数据增强
        val_gen = image_segmentation_generator(
            val_images, val_annotations, val_batch_size,
            n_classes, input_height, input_width, output_height, output_width)
    # 保存该轮次训练完成的模型
    callbacks = [
        CheckpointsCallback(checkpoints_path)
    ]

    if not validate:  # 如果验证集不存在，仅在训练集上训练模型
        model.fit(train_gen, steps_per_epoch=steps_per_epoch,
                  epochs=epochs, callbacks=callbacks)
    else:  # 如果验证集存在，每训练完一个epoch，便在验证集上进行验证
        model.fit(train_gen,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=val_gen,
                  validation_steps=val_steps_per_epoch,
                  epochs=epochs, callbacks=callbacks)
