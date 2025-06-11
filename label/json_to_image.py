# 导入包
import os


# 遍历所有的json文件，转换成图像标签
def json2data():
    root = '/root/Desktop/label_json/'
    root_list = os.listdir(root)
    for i in root_list:
        os.system('labelme_json_to_dataset ' + root + i)


json2data()
