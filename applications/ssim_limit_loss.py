import torch
import ssim_torch_multi
import torch.nn.functional as F

DELTA = 1
BALANCE = 0.00001

class SSIM_limit:
    def __init__(self):
        self.name="SSIM_limit"
    @staticmethod
    def __call__(img1, img2, max=255, binw=None, target_acc_val=0.0, mult=None):
        if binw!=None:
            area_arr = torch.Tensor([1.01, 0.74, 0.03, 0.07, 0.13, 0.07, 0.21, 0.14, 0.5, 0.25, 0.39])
            # area_weight = torch.sum(binw*area_arr)
            ssim = torch.mean(ssim_torch_multi.ssim_torch(img1[0].view(-1,1,32,32),img2[0].view(-1,1,32,32)))
            area = area_arr[mult]
            #print(ssim.item(), img1.isnan().any().item(), img2.isnan().any().item())
            #relu_ssim = F.relu(target_acc_val - ssim)
            n_relu_ssim = F.relu(target_acc_val - ssim)
            

            return - area.data - n_relu_ssim #the right signs are +area-ssim  #area_arr[mult]/n_relu_ssim# + area_weight
        else:
            return torch.mean(ssim_torch_multi.ssim_torch(img1[0].view(-1,1,32,32),img2[0].view(-1,1,32,32)))
