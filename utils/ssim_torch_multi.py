import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

import math
import cv2

def filter3d(img, kernel):
    img_size = img.size()
    # Expose the x dimension and collapse the other 2 dimensions
    img_x = img.permute(0,1,3,2).contiguous().view(-1,1,img_size[2])
    img_x = F.pad(img_x, (5,5), mode="replicate")
    # Perform filtering
    img_x = F.conv1d(img_x, kernel)
    img_x = img_x.view(img_size[0], img_size[1], img_size[3], img_size[2])
    # Expose the y dimension and collapse the other 2 dimensions
    img_y = img_x.permute(0,1,3,2).contiguous().view(-1,1,img_size[3])
    img_y = F.pad(img_y, (5,5), mode="replicate")
    # Perform filtering
    img_y = F.conv1d(img_y, kernel)
    img_y = img_y.view(img_size[0], img_size[1], img_size[2], img_size[3])
    # Expose the z dimension and collapse the other 2 dimensions
    img_z = img_y.permute(0,2,3,1).contiguous().view(-1,1,img_size[1])
    img_z = F.pad(img_z, (5,5), mode="replicate")
    # Perform filtering
    img_z = F.conv1d(img_z, kernel)
    img_z = img_z.view(img_size[0], img_size[2], img_size[3], img_size[1])
    # Permute dimensions back
    img = img_z.permute(0,3,1,2).contiguous()
    return img

def ssim_torch(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.float()
    img2 = img2.float()
    kernel = torch.from_numpy(np.load("../utils/gaussian_kernel.npy").reshape(-1)).float()
    # Reshape kernel for use in 1d conv1d (c_out, c_in, kernel_size)
    kernel = kernel.view(1,1,-1)

    mu1 = filter3d(img1, kernel)
    mu2 = filter3d(img2, kernel)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1*mu2
    sigma1_sq = filter3d(img1**2, kernel) - mu1_sq
    sigma2_sq = filter3d(img2**2, kernel) - mu2_sq
    sigma12 = filter3d(img1*img2, kernel) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean((1,2,3))
