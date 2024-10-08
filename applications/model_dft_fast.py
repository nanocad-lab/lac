import torch
import torch.nn as nn
import numpy as np
import math
import cv2
from torchmetrics import StructuralSimilarityIndexMeasure
import copy
import time
import random
from scipy.linalg import dft
from scipy.fft import dct
import pyximport; pyximport.install()
import mult_approx
import mul_wrapper

class PSNR:
    def __init__(self):
        self.name="PSNR"
    @staticmethod
    def __call__(img1, img2, max=255):
        psnrsum = 0.
        for i in range(img1.size(1)):
            mse = torch.mean((img1[:,i:i+1]-img2[:,i:i+1])**2)
            # return mse
            psnrsum += 20 * torch.log10(255*12*12 / (torch.sqrt(mse) + 1e-6))
        return psnrsum/img1.size(1)

class QuantizeGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
q_50 = torch.Tensor([[16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]])

class Forward_Model(nn.Module):
    
    # Define additional parameters needed
    def __init__(self, size=1000, dtype=torch.float32, mul_approx_func_arr=[1]):
        super(Forward_Model, self).__init__()
        # Define parameters to train
        
        x_re = torch.reshape(torch.Tensor(dft(12).real+0.00001),(1,12*12))#0.5000001 rounds to 1
        x_im = torch.reshape(torch.Tensor(dft(12).imag+0.00001),(1,12*12))

        self.x_re = copy.copy(x_re.data)
        self.x_im = copy.copy(x_im.data)
        self.register_parameter('weight_real', nn.Parameter(torch.from_numpy(np.tile(x_re,(len(mul_approx_func_arr),1)))))
        self.register_parameter('weight_imag', nn.Parameter(torch.from_numpy(np.tile(x_im,(len(mul_approx_func_arr),1)))))
        self.register_parameter('scale', nn.Parameter(torch.from_numpy(np.tile(np.array([1.])*1,(len(mul_approx_func_arr),1)))))
        
        # Define constants
        self.dtype = dtype
        self.size = size
        self.mult_list = mul_approx_func_arr
        self.metric = PSNR()
        self.isNormalized = True
        
        #Save calculated scale
        self.saved_scale = np.tile(np.array([16.]),(len(mul_approx_func_arr),1))
        
    # Define how approximate computation is done
    def forward(self, input, mults=None):
            input = input[:,:,0:12,0:12]

        #Calculate output for all multipliers separately
            output = torch.zeros([len(self.mult_list), 1, self.size,12,12], dtype=self.dtype)
            ii = 0
            for j in self.mult_list: #mul_approx_func_arr:#range(len(mul_approx_func_arr)):
                if (mults is None) or (ii in mults):
                    # Clamping
    #                 with torch.no_grad():
    #                     self.weight.data = torch.clamp(self.weight.data, min=.01, max=255.99)
    #                 weight_in = torch.reshape(torch.stack([torch.clamp(self.weight[j].to(self.dtype), min=0, max=256) for _ in range(self.size)]), (self.size, 1, 3, 3))
                    # Clamping with gradient. Might be the same as normal clamp
                    #weight_clamped = F.hardtanh(self.weight, 0, 255)
                    # Return a rounded version without rounding the master weight
                    #weight_rounded = QuantizeGrad.apply(weight_clamped)
                    #weight_in = torch.reshape(torch.stack([weight_rounded[j].to(self.dtype) for _ in range(self.size)]), (self.size, 1, 3, 3))

                    
                    
    #                 # Approximate convolution
    #                 approx_dct(known_x, input, self.size, j)
                    
                        
                                
    #                 x_data_pre_scale = kernel_multiplier_approx(weight_rounded[j].data, input.data, 1., j)
                    
    #                 # Calculate scale: {approximate output}/{accurate output}
    #                 scale = (torch.mean(x_data_pre_scale)/torch.mean(self.target(input)))
    #                 if scale == 0:
    #                     scale = torch.Tensor([0.001])
    #                 self.saved_scale[j] = scale.data.numpy()
                    
                        #with torch.no_grad():
                    #  self.weight_real.data = torch.clamp(self.weight_real.data, min=-0.999, max=0.999)
                    #  self.weight_imag.data = torch.clamp(self.weight_imag.data, min=-0.999, max=0.999)
                    
                    #weight_clamped = F.hardtanh(self.weight, -1, 1)
                    #weight_clamped = F.hardtanh(self.weight, -1, 1)


                    # Accurate convolution (for gradients)
    #                 mod_weight = weight_in/torch.sum(weight_in) #scale for gradients
                    x = accurate_dft(self.weight_real[ii], self.weight_imag[ii], 12, input, self.size)
                    if j in [1,2,4,10,11]:
                        mod_scale = 65536
                    else:
                        mod_scale = 256
                    # Apply scale and plug gradients
                    x_data_pre_scale = approx_dft(self.weight_real[ii], self.weight_imag[ii], 12, input, self.size, j, mod_scale=mod_scale)
                    x_data = x_data_pre_scale/1.0 #scale for data
                    x.data = x_data
                    
                    output[ii][0] = x
                else:
                    output[ii][0] = torch.zeros(input.size())
                ii+=1
            return output

    # Define how accurate computation should be done
    def target_orig(self, input):
        input = input[:,:,0:12,0:12]
        target = torch.zeros(input.size())
        for i in range(input.size(1)):
            target[0,i] = abs(torch.fft.fft2(input[0,i], norm="backward" ))
        return target
    def target(self, input):
        input = input[:,:,0:12,0:12]
        #target = torch.zeros(input.size())
        #for i in range(input.size(1)):
        #    target[0,i] = abs(torch.fft.fft2(input[0,i]))
        INT_SCALING = 1
        target = abs(approx_dft(self.x_re, self.x_im, 12, input, self.size, -1, mod_scale = 65536))#/INT_SCALING/INT_SCALING
        return target
def approx_dft(x_re, x_im, img_dim, input, size, mult_approx_index, mod_scale = 255):
                output = torch.zeros((1,size,img_dim,img_dim))
                    
                scale = mod_scale
                re_w = torch.round(torch.reshape(x_re, (img_dim, img_dim))* (scale-1));
                im_w = torch.round(torch.reshape(x_im, (img_dim, img_dim))* (scale-1));
                    
                re_temp = torch.zeros((size, img_dim, img_dim))
                im_temp = torch.zeros((size, img_dim, img_dim))
                
                re_out = torch.zeros((size, img_dim, img_dim))
                im_out = torch.zeros((size, img_dim, img_dim))

                # print("re_w")
                # print("mean: ",torch.mean(re_w), " max: ", torch.max(re_w))
                # print("input")
                # print("mean: ",torch.mean(input[0]), " max: ", torch.max(input[0]))
                    
                re_temp = approx_matmul(re_w,input[0],img_dim, mult_approx_index, swap_inputs=True)/scale
                im_temp = approx_matmul(im_w,input[0],img_dim, mult_approx_index, swap_inputs=True)/scale

                # print("re_temp")
                # name = re_temp*scale
                # print("mean: ",torch.mean(name), " max: ", torch.max(name))
                # #print(torch.mean(re_temp))
                    
                #print("")
                re_out = approx_matmul(re_temp,re_w, img_dim, mult_approx_index, swap_inputs=True)
                re_out -= approx_matmul(im_temp,im_w, img_dim, mult_approx_index, swap_inputs=True)
                    
                im_out = approx_matmul(re_temp,im_w, img_dim, mult_approx_index, swap_inputs=True)
                im_out += approx_matmul(im_temp,re_w, img_dim, mult_approx_index, swap_inputs=True)
                    

                re_out = re_out/scale
                im_out = im_out/scale
                output[:] = abs(re_out+im_out*1j)
                return output

def accurate_dft(x_re, x_im, img_dim, input, size):
                output = torch.zeros((1,size,img_dim,img_dim))
                scale = 255
                re_w = torch.reshape(x_re, (img_dim, img_dim)) * scale;
                im_w = torch.reshape(x_im, (img_dim, img_dim)) * scale;
                
                #re_w[0,0].backward()
                    
                # Perform DFT
                #for l in range(size):
                    
                re_temp = torch.zeros((size, img_dim, img_dim))
                im_temp = torch.zeros((size, img_dim, img_dim))
                    
                re_out = torch.zeros((size, img_dim, img_dim))
                im_out = torch.zeros((size, img_dim, img_dim))
                    
                re_temp = torch.matmul(re_w,input[0].to(torch.float))/scale
                im_temp = torch.matmul(im_w,input[0].to(torch.float))/scale
                    
                    
                re_out = torch.matmul(re_temp,re_w)
                re_out -= torch.matmul(im_temp,im_w)
                    
                im_out = torch.matmul(re_temp,im_w)
                im_out += torch.matmul(im_temp,re_w)                    
                    
                re_out = re_out/scale
                im_out = im_out/scale
                output[:] = abs(re_out+im_out*1j)
                    
                return output
            
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