import cv2
import numpy as np
import os

path1 = 'label_png'
count = 1
for file_1 in sorted(os.listdir(path1)):
    img_path = os.path.join(path1, file_1)
    print(count)
    count += 1
    print(img_path)
    path_save = os.path.join('tongue_labels', file_1)
    src_img = cv2.imread(img_path, 0)
    img_res = np.where(src_img == 0, 0, 1)
    cv2.imwrite(path_save, img_res)
