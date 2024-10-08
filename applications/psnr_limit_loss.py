import torch
import ssim_torch_multi
import torch.nn.functional as F

DELTA = 1
BALANCE = 0.00001

class PSNR_limit: #unfinished
    def __init__(self):
        self.name="SSIM_limit"
    @staticmethod
    def __call__(img1, img2, max=255, binw=None, target_acc_val=0.0, mult=None):
        psnrsum = 0.
        for i in range(img1.size(1)):
            mse = torch.mean((img1[:,i:i+1]-img2[:,i:i+1])**2)
            # return mse
            psnrsum += 20 * torch.log10(255*12*12 / (torch.sqrt(mse) + 1e-6))
        psnr_mean = psnrsum/img1.size(1)

        if binw!=None:
            area_arr = torch.Tensor([1.01, 0.74, 0.03, 0.07, 0.13, 0.07, 0.21, 0.14, 0.5, 0.25, 0.39])
            # area_weight = torch.sum(binw*area_arr)
            ssim = torch.mean(ssim_torch_multi.ssim_torch(img1[0].view(-1,1,32,32),img2[0].view(-1,1,32,32)))
            #print(ssim.item(), img1.isnan().any().item(), img2.isnan().any().item())
            n_relu_ssim = F.relu(target_acc_val-ssim)+BALANCE

            return area_arr[mult]/n_relu_ssim# + area_weight
        else:
            return psnr_mean
