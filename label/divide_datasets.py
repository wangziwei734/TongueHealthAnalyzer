import os
import random
import shutil
def moveFile(img_Dir, mask_Dir,train_img_Dir,train_mask_Dir,test_img_Dir,test_mask_Dir):
       img_pathDir = os.listdir(img_Dir)
       filenumber = len(img_pathDir)
       test_rate = 0.1
       test_picknumber = int(filenumber*test_rate)
       sample1 = random.sample(img_pathDir, test_picknumber)
       for test_img in sample1:
           name = test_img.split(".")[0]
           suffix = '.' + test_img.split(".")[1]
           src_img_name1 = img_Dir + name
           dst_img_name1 = test_img_Dir + name
           shutil.move(src_img_name1 + suffix, dst_img_name1 + suffix)
           src_mask_name1 = mask_Dir + name
           dst_mask_name1 = test_mask_Dir + name
           shutil.move(src_mask_name1 + suffix, dst_mask_name1 + suffix)
       img_pathDir = os.listdir(img_Dir)
       for train_img in img_pathDir:
           name = train_img.split(".")[0]
           suffix = '.'+ train_img.split(".")[1]
           src_img_name2 = img_Dir + name
           dst_img_name2 = train_img_Dir + name
           shutil.move(src_img_name2 + suffix, dst_img_name2 + suffix)
           src_mask_name2 = mask_Dir + name
           dst_mask_name2 = train_mask_Dir + name
           shutil.move(src_mask_name2 + suffix, dst_mask_name2 + suffix)
       return
if __name__ == '__main__':
   img_Dir = 'tongue_images/'
   mask_Dir = 'tongue_labels/'
   train_img_Dir = 'tongue_data/train_img/'
   test_img_Dir = 'tongue_data/test_img/'
   train_mask_Dir = 'tongue_data/train_label/'
   test_mask_Dir = 'tongue_data/test_label/'
   if not os.path.exists(train_img_Dir):
       os.makedirs(train_img_Dir)
   if not os.path.exists(test_img_Dir):
       os.makedirs(test_img_Dir)
   if not os.path.exists(train_mask_Dir):
       os.makedirs(train_mask_Dir)
   if not os.path.exists(test_mask_Dir):
       os.makedirs(test_mask_Dir)
   moveFile(img_Dir, mask_Dir,train_img_Dir,train_mask_Dir,test_img_Dir,test_mask_Dir)