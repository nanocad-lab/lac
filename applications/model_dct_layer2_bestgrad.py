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
from scipy.linalg import dft
from scipy.fft import dct
import pyximport; pyximport.install()
import mult_approx
import mul_wrapper_wg as mul_wrapper
#import mul_wrapper

class PSNR:
    def __init__(self):
        self.name="PSNR"
    @staticmethod
    def __call__(img1, img2, max=255):
        psnrsum = 0.
        img1 = img1[0]
        for i in range(img1.size(1)):
            mse = torch.mean((img1[:,i:i+1]-img2[:,i:i+1])**2)
            # return mse
            psnrsum += 20 * torch.log10(127*8*8 / (torch.sqrt(mse) + 1e-6))
        return psnrsum/img1.size(1)

q_50 = torch.Tensor([[16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]])

class QuantizeGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Forward_Model(nn.Module):
    
    # Define additional parameters needed
    def __init__(self, size=1000, dtype=torch.float32, mul_approx_func_arr=[1]):
        super(Forward_Model, self).__init__()
        # Define parameters to train
        dct_mat = dct(np.eye(8), axis=0, norm='ortho')
        weigth_one_multiplier = np.reshape(dct_mat, (1, 64))
        
        self.register_parameter('weight', nn.Parameter(torch.from_numpy(np.tile(weigth_one_multiplier,(len(mul_approx_func_arr),1)))))
        
        self.register_parameter('scale', nn.Parameter(torch.from_numpy(np.tile(np.array([1.])*1,(len(mul_approx_func_arr),1)))))
        self.register_parameter('weight_factor', nn.Parameter(torch.from_numpy(np.tile(np.array([65000.]),(len(mul_approx_func_arr),1)))))
        
        # Define constants
        self.register_buffer('weight_orig', nn.Parameter(torch.from_numpy(weigth_one_multiplier)))
        self.dtype = dtype
        self.size = size
        self.mult_list = mul_approx_func_arr
        self.metric = PSNR()
        self.isNormalized = True
        
        #Save calculated scale
        self.saved_scale = np.tile(np.array([16.]),(len(mul_approx_func_arr),1))
        
    # Define how approximate computation is done
    def forward(self, input, multipliers_list = []):
            if multipliers_list == []:
                 multipliers_list = self.mult_list
                 ii = 0
            else:
                 ii = multipliers_list[0]
                 multipliers_list = [self.mult_list[multipliers_list[0]]]

        #Calculate output for all multipliers separately
            output = torch.zeros([len(multipliers_list), 1, self.size*16,8,8], dtype=self.dtype)
            for j in multipliers_list: #mul_approx_func_arr:#range(len(mul_approx_func_arr)):
                with torch.no_grad():
                    self.weight.data = torch.clamp(self.weight.data, min=-0.999, max=0.999)
                    if j in [1, 2, 4, 8, 11, 10]:
                        self.weight_factor.data = torch.clamp(self.weight_factor.data, min=65536, max=65536)#65534.999
                    else:
                        self.weight_factor.data = torch.clamp(self.weight_factor.data, min=256, max=256)#65534.999
                    
                weight_clamped = F.hardtanh(self.weight, -1, 1)
                weight_factor_clamped = F.hardtanh(self.weight_factor, 1, 65534./2)#65535.
                
                # Return a rounded version without rounding the master weight
                weight_rounded =  self.weight[ii]#*self.weight_factor[ii] #QuantizeGrad.apply(weight_clamped[ii]*weight_factor_clamped[ii])

                #x2 = grad_dct2(weight_rounded, input, self.weight_factor[ii], self.size)
                
                x2 = approx_dct2(torch.unsqueeze(weight_rounded, 0), input, self.weight_factor[ii], self.size, j) #to select manually - self.mult_list[10]
                #print(x_data2.dtype)
                #print(x2.dtype)
                #x2.data = x_data2.to(torch.float32)
                
                if len(multipliers_list) == 1:
                    output[0][0] = x2
                else:
                    output[ii][0] = x2
                ii+=1
            return output

    # Define how accurate computation should be done
    def target(self, input):
        target_layer1 = accurate_dct1(self.weight_orig, input, self.size)
        target_layer2 = accurate_dct2(self.weight_orig, target_layer1, self.size)
        return target_layer2


def approx_dct2(x, input, scale, size, mult_approx_index):
                j = mult_approx_index
                kernel_size = 512
                    
                #scale = 65535
                x_initial = x #torch.Tensor(np.ones((8,8))) #self.weight
                x_adjusted = torch.reshape(x_initial, (8,8))*scale
                x_adjusted_transpose = torch.transpose(x_adjusted, 0, 1)
                 
                temp23 = torch.reshape(mul_wrapper.mul_wrapper(torch.tile(torch.reshape(q_50, (1, 64))[0],(1,16*size)), torch.reshape(input, (1, 64*16*size))[0], j), (16*size, 8,8))+0.00001
                
                return temp23   
                

def accurate_dct2(x, input, size):  
                x_initial = x #torch.Tensor(np.ones((8,8))) #self.weight
                x_adjusted = torch.reshape(x_initial, (8,8)).to(torch.float)
                x_adjusted_transpose = torch.transpose(x_adjusted, 0, 1)    
                temp23 = torch.reshape(torch.mul(torch.tile(torch.reshape(q_50, (1, 64))[0],(1,16*size)), torch.reshape(input, (1, 64*16*size))[0]), (16*size, 8,8))
                return temp23

            
def approx_matmul(a,b,img_dim, mul_approx_index, swap_inputs=False):
    
    # Extend the size of one matrix to fit other matrix
    if a.numel() >= b.numel():
        # Prepare input1 a-input
        in1 = torch.tile(a,(1,img_dim)).view(-1);

        # Prepare input2 b-kernel
        b_trans = torch.reshape(torch.transpose(b.view(img_dim, img_dim),1,0),(1,b.numel()))
        in2  = torch.tile(torch.tile(b_trans,(1,img_dim)),(1,int(a.numel()/(img_dim*img_dim))))
    else:
        # Prepare input1 a-input
        in1 = torch.tile(torch.tile(a,(1,img_dim)).view(-1),(1,int(b.numel()/(img_dim*img_dim))))
        
        
        # Prepare input2 b-kernel
        b_trans = torch.transpose(b.view(int(b.numel()/(img_dim*img_dim)),img_dim, img_dim),1,2)
        in2 = torch.zeros(b.numel()*img_dim)
        for i in range(b_trans.size(0)):
            in2[i*(img_dim**3):(i+1)*(img_dim**3)] = torch.tile(torch.reshape(b_trans[i,:,:],(1,img_dim*img_dim)),(1,img_dim))

    # Perfom approxiamate multiplication (in1 and in2 order matters!)
    if swap_inputs:
        output = mul_wrapper.mul_wrapper(in2,in1,mul_approx_index)
    else:
        output = mul_wrapper.mul_wrapper(in1,in2,mul_approx_index)
        
    # Sum each {img_dim} number of entries
    output_sum = torch.sum((output).view(int(output.numel()/img_dim),img_dim),1)
    
    # Output correct shape
    if a.numel() >= b.numel():
        return output_sum.view(a.size())
    else:
        return output_sum.view(b.size())

def accurate_matmul(a,b,img_dim, mul_approx_index):

    # Prepare input1 a-input
    in1 = torch.tile(a,(1,img_dim)).view(-1);

    # Prepare input2 b-kernel
    b_trans = torch.reshape(torch.transpose(b.view(img_dim, img_dim),1,0),(1,b.numel()))
    in2  = torch.tile(torch.tile(b_trans,(1,img_dim)),(1,int(a.numel()/(img_dim*img_dim))))

    # Perfom approxiamate multiplication
    output = in1*in2

    # Sum each {img_dim} number of entries
    output_sum = torch.sum((output).view(int(output.numel()/img_dim),img_dim),1)
    return output_sum.view(a.size())
