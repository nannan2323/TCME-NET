import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def mse(img1, img2):
    return np.mean ((img1 - img2) ** 2)


def mae(img1, img2):
    return np.mean(np.abs(img1 - img2))


pre_dir = r"/"
gt_dir = r"/"

pre_list = os.listdir (pre_dir)
pre_list.sort ( )

gt_list = os.listdir (gt_dir)
gt_list.sort ( )

final_ssim, final_pnsr, final_mse, final_mae = [], [], [], []
txt_lines = []

for pre_name, gt_name in zip(pre_list, gt_list):
    pre_img = cv2.imread(os.path.join (pre_dir, pre_name), 1)
    gt_img = cv2.imread(os.path.join (gt_dir, gt_name), 1)


    PNSR = np.mean([psnr (gt_img[..., i], pre_img[..., i]) for i in range (3)])
    SSIM = np.mean([ssim (pre_img[..., i], gt_img[..., i]) for i in range (3)])
    MSE = mse(gt_img, pre_img)
    MAE = np.mean([mae (pre_img[..., i], gt_img[..., i]) for i in range (3)])

    final_ssim.append(SSIM)
    final_pnsr.append(PNSR)
    final_mse.append(MSE)
    final_mae.append(MAE)

    line = f"{pre_name}\t\t\t{SSIM:.7f}\t\t{PNSR:.7f}\t\t{MSE:.7f}\t\t{MAE:.7f}\n"
    txt_lines.append(line)

final_average = f"Average\t\t{np.mean (final_ssim):.7f}\t\t{np.mean (final_pnsr):.7f}\t\t{np.mean (final_mse):.7f}\t\t{np.mean (final_mae):.7f}\n"
txt_lines.insert(0, final_average)
txt_lines.insert(1, "Image Name\t\t SSIM \t\t PNSR  \t\t  MSE  \t\t  MAE\n")

with open ("evaluating_indicator.txt", "w") as txt:
    txt.writelines(txt_lines)

print (final_average)
