"""
@author: qiuwenhui
@Software: VSCode
@Time: 2023-03-06 19:02:45
"""

import os
from PIL import Image
import numpy as np

from utils import cvtColor, resize_image
from rich.progress import track


pred_cfg = dict(
    image_shape=(512, 512),
    real_image_dir="../version3_img_new",
    version1_cam_dir="./analyze_relu_layer/version3_img_new_cam_relu_v1",
    version2_cam_dir="./analyze_relu_layer/version3_img_new_cam_relu_v2",
    version3_cam_dir="./analyze_relu_layer/version3_img_new_cam_relu_v3",
    # version1_cam_dir="./version3_img_new_cam_v1",
    # version2_cam_dir="./version3_img_new_cam_v2",
    # version3_cam_dir="./version3_img_new_cam_v3",
    mask_concat_dir="./trainval_mask_concat",
    save_dir="./analyze_relu_layer/version3_img_new_cam_concat_save_v123",
)


def main(pred_cfg):
    real_image_dir, version1_cam_dir, version2_cam_dir, version3_cam_dir = (
        pred_cfg["real_image_dir"],
        pred_cfg["version1_cam_dir"],
        pred_cfg["version2_cam_dir"],
        pred_cfg["version3_cam_dir"],
    )
    save_dir = pred_cfg["save_dir"]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    classes_name = [
        "0_BW",
        "1_HD",
        "2_WR",
        "3_RO",
        "4_RI",
        "5_FV",
        "6_SR",
    ]

    # 生成图片的名称列表
    img_list = [
        img.split(".")[0] for img in os.listdir(real_image_dir) if img.endswith(".jpg")
    ]
    first_dir_path = os.path.join(version1_cam_dir, img_list[0])
    first_img_path = os.path.join(version1_cam_dir, img_list[0], "0_BW")
    classes_name = os.listdir(first_dir_path)
    img_name_list = [
        img.split(".")[0]
        for img in os.listdir(first_img_path)
        if img.split("_")[1] == "model"
    ]

    def sortfun(name):
        num = int(name.split("_")[0])
        return num

    img_name_list.sort(key=sortfun)

    for index, img in enumerate(img_list):
        print(
            f"******************** 🔦🔦🔦 concat {index+1}/{len(img_list)} image ********************"
        )
        real_image = cvtColor(Image.open(os.path.join(real_image_dir, img + ".jpg")))
        real_image, _, _ = resize_image(real_image, (512, 512))
        for cls_index, cls_name in enumerate(classes_name):
            print(f" ⏳⏳⏳ check {cls_index+1}/{len(classes_name)} classes ")
            for img_name in track(img_name_list):
                base_name = os.path.join(img, cls_name, img_name + ".png")
                cam1 = cvtColor(Image.open(os.path.join(version1_cam_dir, base_name)))
                cam2 = cvtColor(Image.open(os.path.join(version2_cam_dir, base_name)))
                cam3 = cvtColor(Image.open(os.path.join(version3_cam_dir, base_name)))
                cam_list = [real_image, cam1, cam2, cam3]
                for i in range(len(cam_list)):
                    cam_list[i] = np.array(cam_list[i])
                concat_img = np.hstack(
                    (
                        cam_list[0],
                        cam_list[1],
                        cam_list[2],
                        cam_list[3],
                    ),
                )

                # concat mask
                mask_path = os.path.join(pred_cfg["mask_concat_dir"], img + ".jpg")
                mask_img = cvtColor(Image.open(mask_path))
                mask_img = np.array(mask_img)
                concat_img = np.vstack((mask_img, concat_img))

                # save img
                save_path = os.path.join(save_dir, img, cls_name)

                def make_dir(path):
                    path_list = path.split("/")[1:]
                    dir_path = "."
                    for i in range(len(path_list)):
                        dir_path = os.path.join(dir_path, path_list[i])
                        if not os.path.exists(dir_path):
                            os.mkdir(dir_path)

                if not os.path.exists(save_path):
                    make_dir(save_path)
                save_concat_img = Image.fromarray(concat_img)
                save_concat_img.save(
                    os.path.join(save_path, img_name + ".png"), quality=95
                )


if __name__ == "__main__":
    main(pred_cfg)
