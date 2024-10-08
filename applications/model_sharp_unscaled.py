import torch
import torch.nn as nn
import numpy as np
import math
import cv2
from torchmetrics import StructuralSimilarityIndexMeasure
import copy
import time
import random
import pyximport; pyximport.install()
import mult_approx
import torch.nn.functional as F
import sys
import mul_wrapper_grad as mul_wrapper
import ssim_limit_loss

sys.path.insert(0, '../utils')
import ssim_torch_multi

def approx_division(value,divisor, verbal=False):
    if divisor <= 0:#NaN bug is usually because of negative divisor
        out = value
    else:
        out = value/(divisor)
        approx_bitshift = torch.round(torch.log2(divisor))
        if(verbal):
            print("DIVISOR",(2**approx_bitshift))
        out.data = value/(2**approx_bitshift)
        #print("div",divisor)
    return out

class SSIM:
    def __init__(self):
        self.name="SSIM"
    @staticmethod
    def __call__(img1, img2, max=255):
        return torch.mean(ssim_torch_multi.ssim_torch(img1[0].view(-1,1,32,32),img2[0].view(-1,1,32,32)))

        # total_ssim = 0
        # temp_ssim = torch.zeros(img1.size()[1])
        # for i,im1 in enumerate(img1[0]):
        #     im1_f = torch.unsqueeze(img1[0,i], 0)
        #     im2_f = torch.unsqueeze(img2[0,i], 0)
        #     ssim = ssim_torch_multi.ssim_torch(im1_f.view(-1,1,32,32),im2_f.view(-1,1,32,32))
        #     total_ssim += ssim
        #     temp_ssim[i] = ssim[0]
        # print("new SSIM",temp_ssim[2])
        # return total_ssim[0]/(img1.size()[1])

class Forward_Model(nn.Module):
    
    # Define additional parameters needed
    def __init__(self, size=1000, dtype=torch.float32, mul_approx_func_arr=[1]):
        super(Forward_Model, self).__init__()
        
        # Define parameters to train
        self.register_parameter('weight', nn.Parameter(torch.from_numpy(np.tile(np.array([-1., -1., -1., -1., 8., -1., -1., -1., -1.])/8,(len(mul_approx_func_arr),1)))))
        self.register_parameter('scale', nn.Parameter(torch.from_numpy(np.tile(np.array([8.])*1,(len(mul_approx_func_arr),1)))))
        self.register_parameter('weight_factor', nn.Parameter(torch.from_numpy(np.tile(np.array([8.])*1,(len(mul_approx_func_arr),1)))))
        
        # Define constants
        self.register_buffer('weight_orig', nn.Parameter(torch.from_numpy(np.array([-1., -1., -1., -1., 8., -1., -1., -1., -1.]))))
        self.dtype = dtype
        self.size = size
        self.mult_list = mul_approx_func_arr
        self.metric = SSIM()
        self.isNormalized = False
        self.limited_metric = ssim_limit_loss.SSIM_limit()
        self.saved_divisor2 = torch.Tensor([0]*(max(self.mult_list)+1))
        
        # Save calculated scale
        self.saved_scale = np.tile(np.array([1.]),(len(mul_approx_func_arr),1))
        
    # Define how approximate computation is done
    def forward(self, input, acc=False, image_size=32, mults=None, use_saved_divisor2=False):
        if acc:
            
            #Never executes
            x = kernel_multiplier_approx(self.weight.to(self.dtype), input.to(self.dtype), self.scale.data)
        else:

        #Calculate output for all multipliers
            output = torch.zeros([len(self.mult_list), 1, self.size,image_size,image_size], dtype=self.dtype)
            ii = 0
            for j in self.mult_list:
                if (mults is None) or (ii in mults):                
                    # Clamping
                    with torch.no_grad():
                        self.weight.data = torch.clamp(self.weight.data, min=-1., max=1.)
                        self.weight_factor.data = torch.clamp(self.weight_factor.data, min=1., max=255.)
    #                 weight_in = torch.reshape(torch.stack([torch.clamp(self.weight[j].to(self.dtype), min=0, max=256) for _ in range(self.size)]), (self.size, 1, 3, 3))
                    # Clamping with gradient. Might be the same as normal clamp
                    
                    weight_clamped = F.hardtanh(self.weight, -1, 1)
                    weight_factor_clamped = F.hardtanh(self.weight_factor, 1, 255)
                    
                    weight_rounded = weight_clamped[ii]*weight_factor_clamped[ii]
                    
                    weight_in = torch.reshape(torch.stack([weight_rounded for _ in range(self.size)]), (self.size, 1, 3, 3))
                    
                    # Approxiamte convoluiton
                    x_data = approx_sharp(self, weight_rounded, input, self.size, j, 1, use_saved_divisor2=use_saved_divisor2, save_divisor2=(not use_saved_divisor2))
                    #print("x_data",x_data.isnan().any())
                    # Apply scale for gradients
                    #scale_x = torch.sum(torch.abs(weight_in))
                    #if scale_x == 0:
                    #    scale_x = 0.0000001
                    #mod_weight = weight_in/scale_x #scale for gradients
                    
                    # Accurate convolution (for gradients)
                    
                    #x = grad_sharp(weight_in, input, self.size,self.weight_factor[ii].data)
                    
                    # Apply scale to data
                    #scale = (torch.mean(x_data_pre_scale)/torch.mean(self.target(input)))
                    #self.saved_scale[j] = scale.data.numpy()
                    #x_data = x_data_pre_scale/1.0 #scale #scale for data
                    
                    # Plug in gradients
                    #x.data = x_data
                    
                    output[ii][0] = x_data
                else:
                    output[ii][0] = torch.zeros(input.size())
                ii+=1
        return output

    # Define how accurate computation should be done
    def target(self, input):
        weight_in = torch.reshape(self.weight_orig.to(self.dtype), (1, 1, 3, 3))*8

        x = approx_sharp(self, weight_in, input, self.size, -1, 1, const_divisor=True)#, use_saved_divisor2=False, istarget=True)
        #x = nn.functional.conv2d(input.to(self.dtype), weight=Scale_Grad.apply(weight_in.to(self.dtype), lr_scale[j]), padding=(1,1), groups=input.size(1))  
        return x

def accurate_sharp(x, input, size):
    target1 = nn.functional.conv2d(input.to(torch.float), weight=x.to(torch.float), padding=(1,1), groups=input.size(1))
    result = torch.zeros(input.shape)
    # for i in range(size):
    #     target = target1[:,i:i+1]
    #     if (torch.max(target)-torch.min(target)) == 0:
    #         mod = ((target-torch.min(target))/(0.001))*255.
    #     else:
    #         mod = ((target-torch.min(target))/(torch.max(target)-torch.min(target)))*255.
    #     sharpened = input[:,i:i+1] + mod #sharpened
    #     if torch.max(sharpened)-torch.min(sharpened) == 0:
    #         result[0,i:i+1] = ((sharpened-torch.min(sharpened))/(0.001))*255.
    #     else:
    #         result[0,i:i+1] = ((sharpened-torch.min(sharpened))/(torch.max(sharpened)-torch.min(sharpened)))*255.
    return input + target1

# def accurate_sharp(x, input, size, div_factor=1.0):
#     target1 = nn.functional.conv2d(input.to(torch.float), weight=x.to(torch.float), padding=(1,1), groups=input.size(1))
#     return input+target1/div_factor

def grad_sharp(x, input, size, div_factor):
    target1 = nn.functional.conv2d(input.to(torch.float), weight=x.to(torch.float), padding=(1,1), groups=input.size(1))
    return input+target1/div_factor

def approx_sharp(self, x, input, size, mult_approx_index, div_factor, use_saved_divisor2=False, save_divisor2=False, const_divisor=False):
    j = mult_approx_index
    result = kernel_multiplier_approx(x, input.data, 1., j)
    #print("result",result.isnan().any())
    div = torch.max(result)/128.
    if use_saved_divisor2:
        div = self.saved_divisor2[mult_approx_index]#torch.Tensor([self.saved_divisor2])[0]
    if save_divisor2:
        self.saved_divisor2[mult_approx_index] = div.data.item()
    if const_divisor:
        div = torch.Tensor([128.])[0]
    return input+approx_division(result, div, verbal=False)

def kernel_multiplier_approx(tker1, x, scale, mul_approx_index):
    tker = tker1.view((3,3))
    
    kernel_size = tker.shape
    kernel_elems = kernel_size[0]*kernel_size[1]

    #unfolding
    unfold = nn.Unfold(kernel_size=kernel_size, padding=(1,1))
    output = unfold(x)

    #prepare weights for multiplication
    ort = tker.view(tker.numel(), -1).to(float)
    amount = x.numel()
    l = output.shape[2]
    in1 = torch.stack([ort for _ in range(l)], dim=2)
    in2 = torch.stack([in1 for _ in range(x.shape[1])], dim=0)

    #multiply in parallel
    #Accurate version: out = in2.view(1,kernel_elems*amount)*output.view(1,kernel_elems*amount)
    #Approximate version: out = mult_approx.mult_approx(output.view(1,output.numel()), in2.view(1,output.numel()), mul_approx_index)
    out = mul_wrapper.mul_wrapper(output.view(1,output.numel()), in2.view(1,output.numel()), mul_approx_index)
    
    #fold the output of multiplication
    resized = out.view(x.shape[1],kernel_elems, int(out.numel()/(kernel_elems*x.shape[1])))
    
    #shape the output
    summed = torch.sum(resized,1)
    output_size = x.shape[2] - kernel_size[1]+3
    return summed.view(1, x.shape[1],output_size,output_size)/scale