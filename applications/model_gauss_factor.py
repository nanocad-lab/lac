import torch
import torch.nn as nn
import numpy as np
import math
import cv2
from torchmetrics import StructuralSimilarityIndexMeasure
import copy
import time
import random
import torch.nn.functional as F
import pyximport; pyximport.install()
import mult_approx
import sys
import mul_wrapper
import ssim_limit_loss

sys.path.insert(0, '../utils')
import ssim_torch_multi

class QuantizeGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.clamp(torch.round(x),0.001,None)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
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
        self.register_parameter('weight', nn.Parameter(torch.from_numpy(np.tile(np.array([1., 2., 1., 2., 4., 2., 1., 2., 1.])/16,(len(mul_approx_func_arr),1)))))
        self.register_parameter('scale', nn.Parameter(torch.from_numpy(np.tile(np.array([16.])*1,(len(mul_approx_func_arr),1)))))
        self.register_parameter('weight_factor', nn.Parameter(torch.from_numpy(np.tile(np.array([16.]),(len(mul_approx_func_arr),1)))))
        
        # Define constants
        self.register_buffer('weight_orig', nn.Parameter(torch.from_numpy(np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]]))))
        self.dtype = dtype
        self.size = size
        self.mult_list = mul_approx_func_arr
        self.metric = SSIM()
        self.limited_metric = ssim_limit_loss.SSIM_limit()
        self.isNormalized = False
        
        #Save calculated scale
        self.saved_scale = np.tile(np.array([16.]),(len(mul_approx_func_arr),1))
        
    # Define how approximate computation is done
    def forward(self, input, acc=False, image_size=32, mults=None):
        if acc:
            
            #Never executes
            x = kernel_multiplier_approx(self.weight.to(self.dtype), input.to(self.dtype), self.scale.data)
        else:

        #Calculate output for all multipliers separately
            output = torch.zeros([len(self.mult_list), 1, self.size,image_size,image_size], dtype=self.dtype)
            ii=0
            for j in self.mult_list:
                
                if (mults is None) or (ii in mults):
                    # Clamping
                    with torch.no_grad():
                        self.weight.data = torch.clamp(self.weight.data, min=.001, max=0.999)
                        self.weight_factor.data = torch.clamp(self.weight_factor.data, min=1.001, max=254.999)
    #                 weight_in = torch.reshape(torch.stack([torch.clamp(self.weight[j].to(self.dtype), min=0, max=256) for _ in range(self.size)]), (self.size, 1, 3, 3))
                    # Clamping with gradient. Might be the same as normal clamp
                    
                    weight_clamped = F.hardtanh(self.weight, 0, 1)
                    weight_factor_clamped = F.hardtanh(self.weight_factor, 1, 255)
                    # Return a rounded version without rounding the master weight
                    weight_rounded = QuantizeGrad.apply(weight_clamped[ii]*weight_factor_clamped[ii])
                    #weight_rounded = weight_clamped[ii]*weight_factor_clamped[ii]
                    #weight_rounded[ii] = weight_rounded[ii]*self.weight_factor[ii]
                    weight_in = torch.reshape(torch.stack([weight_rounded for _ in range(self.size)]), (self.size, 1, 3, 3))
                    
                    # Approximate convolution
    #                 x_data_pre_scale = kernel_multiplier_approx(self.weight[j].data, input.data, 1., j)
                    x_data_pre_scale = kernel_multiplier_approx((weight_clamped[ii]*weight_factor_clamped[ii]).data, input.data, 1., j)
                    
                    # Calculate scale: {approximate output}/{accurate output}
                    #scale = (torch.mean(x_data_pre_scale)/torch.mean(self.target(input)))
                    #if scale == 0:
                    #    scale = torch.Tensor([0.001])
                    #self.saved_scale[ii] = scale.data.numpy()
                    
                    # Accurate convolution (for gradients)
                    mod_weight = weight_in #/torch.sum(weight_rounded[ii]) #scale for gradients
                    x = nn.functional.conv2d(input.to(self.dtype), weight=mod_weight.to(self.dtype), padding=(1,1), groups=input.size(1))
                    
                    # Apply scale and plug gradients
                    x_data = x_data_pre_scale#/self.scale[ii] #scale for data
                    x.data = x_data
                    output[ii][0] = x
                else:
                    output[ii][0] = torch.zeros(input.size())
                ii+=1
        return output

    # Define how accurate computation should be done
    def target(self, input):
        weight_in = torch.reshape(torch.stack([self.weight_orig.to(self.dtype) for _ in range(self.size)]), (self.size, 1, 3, 3))
        output1 = nn.functional.conv2d(input.to(self.dtype), weight=weight_in, padding=(1,1), groups=input.size(1))
        return output1

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