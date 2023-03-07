"""
@author: qiuwenhui
@Software: VSCode
@Time: 2023-03-05 17:22:03
"""
import os
from PIL import Image
import numpy as np

from utils import cvtColor, resize_image
from rich.progress import track


pred_cfg = dict(
    image_shape=(512, 512),
    real_mask_dir="./val_mask",
    version1_hybrid_dir="./val_img_version1",
    version2_hybrid_dir="./val_img_version2",
    version3_hybrid_dir="./val_img_version3",
    save_dir="./trainval_mask_concat_512",
)


def main(pred_cfg):
    real_mask_dir, version1_hybrid_dir, version2_hybrid_dir, version3_hybrid_dir = (
        pred_cfg["real_mask_dir"],
        pred_cfg["version1_hybrid_dir"],
        pred_cfg["version2_hybrid_dir"],
        pred_cfg["version3_hybrid_dir"],
    )
    save_dir = pred_cfg["save_dir"]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    img_list = [
        img.split(".")[0] for img in os.listdir(real_mask_dir) if img.endswith("png")
    ]
    image_shape = pred_cfg["image_shape"]

    for index, img in enumerate(track(img_list)):
        real_mask = cvtColor(Image.open(os.path.join(real_mask_dir, img + ".png")))
        hybrid1 = cvtColor(Image.open(os.path.join(version1_hybrid_dir, img + ".jpg")))
        hybrid2 = cvtColor(Image.open(os.path.join(version2_hybrid_dir, img + ".jpg")))
        hybrid3 = cvtColor(Image.open(os.path.join(version3_hybrid_dir, img + ".jpg")))
        mask_list = [real_mask, hybrid1, hybrid2, hybrid3]

        # new_mask_list = mask_list
        new_mask_list = []
        for mask in mask_list:
            new_mask, _, _ = resize_image(mask, image_shape)
            new_mask_list.append(new_mask)

        for i in range(len(new_mask_list)):
            new_mask_list[i] = np.array(new_mask_list[i])

        concat_img = np.hstack(
            (new_mask_list[0], new_mask_list[3], new_mask_list[2], new_mask_list[1]),
        )
        save_concat_img = Image.fromarray(concat_img)
        save_concat_img.save(os.path.join(save_dir, img + ".jpg"), quality=95)


if __name__ == "__main__":
    main(pred_cfg)
