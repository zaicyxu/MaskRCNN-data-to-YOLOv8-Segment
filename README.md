# MaskRCNN-data-to-YOLOv8-Segment
MaskRCNN data to YOLOv8 Segment
## MSCOCO to TXT

1. there has a offical website by ultralytics give some method turn the coco into txt:

https://github.com/ultralytics/JSON2YOLO

1. After the format conversion, the data is not divided, and the pictures are not matched with the corresponding txt files, so the following code is needed for processing:

```python
import os
import shutil

def filter_files(img_folder, txt_folder, output_img_folder, output_txt_folder):
    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(output_txt_folder, exist_ok=True)

    img_files = {os.path.splitext(f)[0] for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))}
    txt_files = {os.path.splitext(f)[0] for f in os.listdir(txt_folder) if os.path.isfile(os.path.join(txt_folder, f))}

    common_files = img_files.intersection(txt_files)

    for file in common_files:
        img_file_path = os.path.join(img_folder, file + '.png')  # 假设图片扩展名是 .png，可以根据需要修改
        if not os.path.exists(img_file_path):
            img_file_path = os.path.join(img_folder, file + '.jpg')  
        if os.path.exists(img_file_path):
            shutil.copy(img_file_path, os.path.join(output_img_folder, os.path.basename(img_file_path)))

        txt_file_path = os.path.join(txt_folder, file + '.txt')
        if os.path.exists(txt_file_path):
            shutil.copy(txt_file_path, os.path.join(output_txt_folder, os.path.basename(txt_file_path)))

if __name__ == "__main__":
    img_folder = r"D:\work\JSON2YOLO-main\荧光数据\JPEGImages"
    txt_folder = r"D:\work\JSON2YOLO-main\new_dir\labels\val"
    output_img_folder = r"D:\work\JSON2YOLO-main\new_dir\labels\linshi\img"
    output_txt_folder = r"D:\work\JSON2YOLO-main\new_dir\labels\linshi\text"

    filter_files(img_folder, txt_folder, output_img_folder, output_txt_folder)
```

1. The data set is divided according to the data requirements of the ultralytics HUB website. If there is no code for dividing the data set, you can refer to the following code:

```python
import os
import shutil
import random

def split_dataset(img_folder, txt_folder, output_folder, train_ratio=0.8):
    train_img_folder = os.path.join(output_folder, 'train', 'image')
    train_label_folder = os.path.join(output_folder, 'train', 'label')
    val_img_folder = os.path.join(output_folder, 'val', 'image')
    val_label_folder = os.path.join(output_folder, 'val', 'label')

    os.makedirs(train_img_folder, exist_ok=True)
    os.makedirs(train_label_folder, exist_ok=True)
    os.makedirs(val_img_folder, exist_ok=True)
    os.makedirs(val_label_folder, exist_ok=True)

    img_files = {os.path.splitext(f)[0] for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))}
    txt_files = {os.path.splitext(f)[0] for f in os.listdir(txt_folder) if os.path.isfile(os.path.join(txt_folder, f))}

    common_files = list(img_files.intersection(txt_files))

    random.shuffle(common_files)

    train_size = int(len(common_files) * train_ratio)

    # 划分训练集和验证集
    train_files = common_files[:train_size]
    val_files = common_files[train_size:]

    for file in train_files:
        img_file_path = os.path.join(img_folder, file + '.png')
        if not os.path.exists(img_file_path):
            img_file_path = os.path.join(img_folder, file + '.jpg')
        if os.path.exists(img_file_path):
            shutil.copy(img_file_path, os.path.join(train_img_folder, os.path.basename(img_file_path)))

        txt_file_path = os.path.join(txt_folder, file + '.txt')
        if os.path.exists(txt_file_path):
            shutil.copy(txt_file_path, os.path.join(train_label_folder, os.path.basename(txt_file_path)))

    for file in val_files:
        img_file_path = os.path.join(img_folder, file + '.png')
        if not os.path.exists(img_file_path):
            img_file_path = os.path.join(img_folder, file + '.jpg')
        if os.path.exists(img_file_path):
            shutil.copy(img_file_path, os.path.join(val_img_folder, os.path.basename(img_file_path)))

        txt_file_path = os.path.join(txt_folder, file + '.txt')
        if os.path.exists(txt_file_path):
            shutil.copy(txt_file_path, os.path.join(val_label_folder, os.path.basename(txt_file_path)))

if __name__ == "__main__":
    img_folder = r"C:\Users\a\Desktop\segment\images"
    txt_folder = r"C:\Users\a\Desktop\segment\labels"
    output_folder = r"C:\Users\a\Desktop\segment\set"

    split_dataset(img_folder, txt_folder, output_folder, train_ratio=0.8)
```

## Ultralytics HUB

Website: http://hub.ultralytics.com/home

Tips: There are two points to note when using local computing power:

当出现错误：UsageError: api_key not configured (no-tty). call wand，是安装了wandb包，但是没有使用二次验证，直接登陆云端账户，所以需要

```python
pip uninstall wandb
```

1. 按照官网给出的代码运行会由于多进程问题报错，需要把运行代码改为：

```python
from ultralytics import YOLO, checks, hub

if __name__ == '__main__':
    checks()
    # 以下两段代码会由于模型任务和id的不同链接地址发生变化。
    # hub.login('84b31a85e6f153936e8a3ab2b7cbef06e1be441fb7')
    # model = YOLO('https://hub.ultralytics.com/models/z7oSiCvRKS2lTqOypQay')
    results = model.train()
```
