import os

img_Dir = 'tongue_images'
mask_Dir = 'tongue_labels'
for src_file in os.listdir(img_Dir):
    src_name = src_file.split(".")[0]
    suffix = '.' + src_file.split(".")[1]
    for mask_file in os.listdir(mask_Dir):
        mask_name = mask_file.split(".")[0]
        if src_name == mask_name:
            os.rename(os.path.join(mask_Dir, mask_file), os.path.join(mask_Dir, mask_name + suffix))
