"""
@author: qiuwenhui
@Software: VSCode
@Time: 2023-03-06 21:17:13
"""
import os


def main():
    # ******************** 目标文件夹路径 ********************
    root = "./analyze_relu_layer/version3_img_new_cam_relu_v1"
    # root = "./version3_img_new_cam_v1"
    for dir in os.listdir(root):
        class_path = os.path.join(root, dir)
        for cls_name in os.listdir(class_path):
            file_path = os.path.join(class_path, cls_name)
            for filename in os.listdir(file_path):
                if "cat_conv2" in filename:
                    old_1 = "cat_conv2"
                    new_1 = "cat_conv3"
                    old_name1 = os.path.join(file_path, filename)
                    new_name1 = old_name1.replace(old_1, new_1)
                    os.rename(old_name1, new_name1)
                    print(f"{old_name1} **********> {new_name1}")
                elif "conv0_shortcut" in filename:
                    old_2 = "conv0_shortcut"
                    new_2 = "conv1_shortcut"
                    old_name2 = os.path.join(file_path, filename)
                    new_name2 = old_name2.replace(old_2, new_2)
                    os.rename(old_name2, new_name2)
                    print(f"{old_name2} ----------> {new_name2}")

    # os.rename("./65_model_cat_conv3.png", "./123.png")


if __name__ == "__main__":
    main()
