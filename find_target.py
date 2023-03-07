"""
@author: qiuwenhui
@Software: VSCode
@Time: 2023-03-06 18:33:10
"""


import os
import shutil
from rich.progress import track


def main():
    target_concat_train = "./train_mask_concat"
    target_concat_val = "./val_mask_concat"
    target_mask = "../version3_img"
    target_mask_list = [img for img in os.listdir(target_mask) if img.endswith(".jpg")]
    print(target_mask_list)

    train_concat = os.listdir(target_concat_train)
    val_concat = os.listdir(target_concat_val)

    save_dir = "./version3_img_concat"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for img in track(target_mask_list):
        if img in train_concat:
            shutil.copy(os.path.join(target_concat_train, img), save_dir)
        elif img in val_concat:
            shutil.copy(os.path.join(target_concat_val, img), save_dir)
        else:
            print(f"{img} concat img do not exist")


if __name__ == "__main__":
    main()
