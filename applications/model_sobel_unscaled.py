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

np.set_printoptions(linewidth=np.inf)
sys.path.insert(0, '../utils')
import ssim_torch_multi
import mul_wrapper
import ssim_limit_loss

class SSIM:
    def __init__(self):
        self.name="SSIM"
    @staticmethod
    def __call__(img1, img2, max=255):
        return torch.mean(ssim_torch_multi.ssim_torch(img1[0].view(-1,1,32,32),img2[0].view(-1,1,32,32)))

class Forward_Model(nn.Module):
    
    # Define additional parameters needed
    def __init__(self, size=1000, dtype=torch.float32, mul_approx_func_arr=[1]):
        super(Forward_Model, self).__init__()
        
        # Define parameters to train
        self.register_parameter('weight', nn.Parameter(torch.from_numpy(np.tile(np.array([1., 0., -1., 2., 0., -2., 1., 0., -1.])/2,(len(mul_approx_func_arr),1)))))
        self.register_parameter('scale', nn.Parameter(torch.from_numpy(np.tile(np.array([1.])*1,(len(mul_approx_func_arr),1)))))
        self.register_parameter('weight_factor', nn.Parameter(torch.from_numpy(np.tile(np.array([1.])*2,(len(mul_approx_func_arr),1)))))
        
        # Define constants
        self.register_buffer('weight_orig', nn.Parameter(torch.from_numpy(np.array([1., 0., -1., 2., 0., -2., 1., 0., -1.]))))
        self.dtype = dtype
        self.size = size
        self.mult_list = mul_approx_func_arr
        self.metric = SSIM()
        self.isNormalized = False
        self.limited_metric = ssim_limit_loss.SSIM_limit()
        
        # Save calculated scale
        self.saved_scale = np.tile(np.array([1.]),(len(mul_approx_func_arr),1))
        
    # Define how approximate computation is done
    def forward(self, input, acc=False, image_size=32, mults=None):
        if acc:
            pass
            #Never executes
        else:

            image_size=input.size(-1)
        #Calculate output for all multipliers
            output = torch.zeros([len(self.mult_list), 1, self.size,image_size,image_size], dtype=self.dtype)
            ii = 0
            for j in self.mult_list:
                if (mults is None) or (ii in mults):                
                # Clamping
                # Clamping
                    with torch.no_grad():
                        self.weight.data = torch.clamp(self.weight.data, min=-1, max=1)
                        self.weight_factor.data = torch.clamp(self.weight_factor.data, min=1.001, max=254.999)
    #                 weight_in = torch.reshape(torch.stack([torch.clamp(self.weight[j].to(self.dtype), min=0, max=256) for _ in range(self.size)]), (self.size, 1, 3, 3))
                    # Clamping with gradient. Might be the same as normal clamp
                    
                    weight_clamped = F.hardtanh(self.weight, -1, 1)
                    weight_factor_clamped = F.hardtanh(self.weight_factor, 1, 255)
                        
                    weight_rounded = weight_clamped[ii]*weight_factor_clamped[ii]
                    
                    weight_in = torch.reshape(torch.stack([weight_rounded for _ in range(self.size)]), (self.size, 1, 3, 3))
                    
                    # Approxiamte convoluiton
                    
                    x_data = approx(weight_rounded, input, self.size, j)
                    # Apply scale for gradients
                    #scale_x = torch.sum(torch.abs(weight_in))
                    #if scale_x == 0:
                    #    scale_x = 0.0000001
                    #mod_weight = weight_in/scale_x #scale for gradients
                    
                    # Accurate convolution (for gradients)
                    x = accurate_grad(weight_in*weight_factor_clamped[ii], input, self.size, image_size=image_size)
                    
                    # Apply scale to data
                    #scale = (torch.mean(x_data_pre_scale)/torch.mean(self.target(input)))
                    #self.saved_scale[j] = scale.data.numpy()
                    #x_data = x_data_pre_scale/1.0 #scale #scale for data
                    
                    # Plug in gradients
                    x.data = x_data
                    
                    output[ii][0] = x
                else:
                    output[ii][0] = torch.zeros(input.size())
                ii+=1
        return output

    # Define how accurate computation should be done
    def target(self, input):
        weight_in = torch.reshape(self.weight_orig.to(self.dtype), (1, 1, 3, 3))
        image_size = input.size(-1)
        x = approx(weight_in, input, self.size, -1)
        #x = nn.functional.conv2d(input.to(self.dtype), weight=Scale_Grad.apply(weight_in.to(self.dtype), lr_scale[j]), padding=(1,1), groups=input.size(1))  
        return x

def accurate_target(x, input, size, image_size=32):
    result = torch.zeros(1,size,image_size,image_size)
    x_trans = torch.transpose(x,2,3)
    gx = nn.functional.conv2d(input.to(torch.float), weight=x.to(torch.float), padding=(1,1), groups=input.size(1))
    gy = nn.functional.conv2d(input.to(torch.float), weight=x_trans.to(torch.float), padding=(1,1), groups=input.size(1))
    g = abs(gx) + abs(gy)
    # for i in range(size):
    #     g = abs(gx[0,i:i+1]) + abs(gy[0,i:i+1])
    #     gmin = torch.min(g)
    #     dx = torch.max(g)-gmin
    #     if dx == 0:
    #         dx = 1.
    #     result[0,i:i+1] =(g-gmin)/dx*255
    #print(result)
    return g

def accurate_grad(x, input, size, image_size=32):
    result = torch.zeros(1,size,image_size,image_size)
    x_trans = torch.transpose(x,2,3)
    gx = nn.functional.conv2d(input.to(torch.float), weight=x.to(torch.float), padding=(1,1), groups=input.size(1))
    gy = nn.functional.conv2d(input.to(torch.float), weight=x_trans.to(torch.float), padding=(1,1), groups=input.size(1))
    g = abs(gx) + abs(gy)
    return g

def approx(x, input, size, mult_approx_index):
    j = mult_approx_index
    x = torch.reshape(x, (3,3))
    result = torch.zeros(input.shape)
    x_trans = torch.transpose(x,0,1)
    gx = kernel_multiplier_approx(x, input.data, 1.0, mult_approx_index)
    gy = kernel_multiplier_approx(x_trans, input.data, 1.0, mult_approx_index)
    g = abs(gx) + abs(gy)
    #print("min: {0}, mean: {1}, max: {2}".format(torch.min(g), torch.mean(g), torch.max(g)))
    return g

def kernel_multiplier_approx(tker1, x, scale, mul_approx_index):
    tker = tker1.view((3,3))
    
    kernel_size = tker.shape
    kernel_elems = kernel_size[0]*kernel_size[1]

    #unfolding
    unfold = nn.Unfold(kernel_size=kernel_size, padding=(1,1))
    output = unfold(x)

    #prepare weights for multiplication
    ort = torch.reshape(tker,(tker.numel(), -1)).to(float)
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
