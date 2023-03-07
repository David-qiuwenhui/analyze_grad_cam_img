"""
@author: qiuwenhui
@Software: VSCode
@Time: 2023-03-06 20:27:10
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
    save_dir="./stack_concat",
)


def main(pred_cfg):
    real_mask_dir, version1_hybrid_dir, = (
        pred_cfg["real_mask_dir"],
        pred_cfg["version1_hybrid_dir"],
    )
    save_dir = pred_cfg["save_dir"]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    img_list = [
        img.split(".")[0] for img in os.listdir(real_mask_dir) if img.endswith("png")
    ]

    for index, img in enumerate(track(img_list)):
        real_mask = cvtColor(Image.open(os.path.join(real_mask_dir, img + ".png")))
        hybrid1 = cvtColor(Image.open(os.path.join(version1_hybrid_dir, img + ".jpg")))
        mask_list = [real_mask, hybrid1]

        new_mask_list = mask_list
        # new_mask_list = []
        # for mask in mask_list:
        #     new_mask, _, _ = resize_image(mask, image_shape)
        #     new_mask_list.append(new_mask)

        for i in range(len(new_mask_list)):
            new_mask_list[i] = np.array(new_mask_list[i])

        concat_img = np.vstack(
            (new_mask_list[0], new_mask_list[1]),
        )
        save_concat_img = Image.fromarray(concat_img)
        save_concat_img.save(os.path.join(save_dir, img + ".png"), quality=95)


if __name__ == "__main__":
    main(pred_cfg)
