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
import torch.nn.functional as F
import mul_wrapper_fixed as mul_wrapper

#Define PSNR loss
class RelativeError:
    def __init__(self):
        self.name="RelativeError"
    @staticmethod
    def __call__(out,target, max=255):
        out = torch.squeeze(out)
        target = torch.squeeze(target)
        sume = 0
        for i in range(out.size(0)):
            diff1 = target[i,0]-out[i,0]
            diff2 = target[i,1]-out[i,1]
            num = torch.sqrt(diff1**2+diff2**2)
            den = torch.sqrt(target[i,0]**2+target[i,1]**2)
            if den == 0 or math.isnan(num) or math.isnan(den):
                e = 1.0
            else:
                e = num/den
            sume += e
        return -sume/out.size(0)

l1 = torch.Tensor([0.5])
l2 = torch.Tensor([0.5])

class QuantizeGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.clamp(torch.round(x),0.001,None)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Forward_Model(nn.Module):
    
    # Define additional parameters needed
    def __init__(self, size=1000, dtype=torch.float32, mul_approx_func_arr=[1]):
        super(Forward_Model, self).__init__()
        
        # Define parameters to train
        self.register_parameter('weight', nn.Parameter(torch.from_numpy(np.tile(np.array([1.,1.,1.,1.])*100,(len(mul_approx_func_arr),1)))))
        #self.register_parameter('weight_factor', nn.Parameter(torch.from_numpy(np.tile(np.array([1.])*200,(len(mul_approx_func_arr),1)))))
        
        # Define constants
        self.dtype = dtype
        self.size = size
        self.mult_list = mul_approx_func_arr
        self.metric = RelativeError()
        self.isNormalized = True

    # Define how approximate computation is done
    def forward(self, input, acc=False, mults=None):

        #Calculate output for all multipliers
            output = torch.zeros([len(self.mult_list), 1, self.size, 2], dtype=self.dtype)
            ii = 0
            for j in self.mult_list:
                if (mults is None) or (ii in mults):                  
                    # Clamping
                    #with torch.no_grad():
                    #    self.weight.data = torch.clamp(self.weight.data, min=-0.999, max=0.999)
                    #    self.weight_factor.data = torch.clamp(self.weight_factor.data, min=1.001, max=254.999)

                    # Clamping with gradient. Might be the same as normal clamp
                    #weight_clamped = F.hardtanh(self.weight, -1, 1)
                    #weight_factor_clamped = F.hardtanh(self.weight_factor, 1, 255)
                    #with torch.no_grad():
                    #    self.weight_factor.data = torch.clamp(self.weight_factor.data, min=0.5, max=127.999)
                    #    self.weight.data = torch.clamp(self.weight.data, min=-0.999, max=0.999)
                    #factor = F.hardtanh(self.weight_factor[ii], 0.49, 128)
                    #weight = F.hardtanh(self.weight[ii], -258, 258)
                    
                    weight_scaled = QuantizeGrad.apply(self.weight[ii]) #QuantizeGrad.apply(weight*factor)
                    #weight_scaled = self.weight[ii]
                    # Approxiamte convoluiton
                    x_data = approx(weight_scaled, input, self.size, j)
                    
                    # Accurate convolution (for gradients)
                    x = grad(weight_scaled, input, self.size)
                    
                    # Plug in gradients
                    x.data = x_data
                    
                    output[ii][0] = x
                else:
                    output[ii][0] = torch.zeros(input.size())
                ii+=1
            return output

    # Define how accurate computation should be done
    def target(self, input):
        x = torch.unsqueeze(accurate(input, self.size),0)
        return x

def approx(c, input, size, mul_approx_index):
    result = torch.zeros((1,size,2))
    for i in range(size):
        x = input[0,i,0]
        y = input[0,i,1]
        
        x_scaled = x*c[0]
        y_scaled = y*c[1]
        l1_scaled = l1*c[2]
        l2_scaled = l2*c[3]
        
        dis = mul_wrapper.mul_wrapper(x_scaled,x_scaled,mul_approx_index)+mul_wrapper.mul_wrapper(y_scaled,y_scaled,mul_approx_index)+0.000001
        theta2 = torch.acos(torch.clamp((dis-mul_wrapper.mul_wrapper(l1_scaled,l1_scaled,mul_approx_index)-mul_wrapper.mul_wrapper(l2_scaled,l2_scaled,mul_approx_index))/(2*mul_wrapper.mul_wrapper(l1_scaled,l2_scaled,mul_approx_index)+0.000001),min = -1, max = 1))
        theta1 = torch.asin(torch.clamp((mul_wrapper.mul_wrapper(y_scaled,(l1_scaled + l2_scaled * torch.cos(theta2)),mul_approx_index) - mul_wrapper.mul_wrapper(x_scaled,l2_scaled,mul_approx_index) * torch.sin(theta2))/dis,min = -1, max = 1))

        result[0,i,0] = theta1
        result[0,i,1] = theta2
    return result
    
def grad(c, input, size):
    result = torch.zeros((1,size,2))
    for i in range(size):
        x = input[0,i,0]
        y = input[0,i,1]
        
        x_scaled = x*c[0]
        y_scaled = y*c[1]
        l1_scaled = l1*c[2]
        l2_scaled = l2*c[3]
        
        arg2 = ((x_scaled*x_scaled)+(y_scaled*y_scaled)-(l1_scaled*l1_scaled)-(l2_scaled*l2_scaled))/(2*l1_scaled*l2_scaled+1)
        with torch.no_grad():
                arg2.data = torch.clamp(arg2.data, min=-0.999, max=0.999)
        arg2_clamped = F.hardtanh(arg2, -1, 1)
        
        theta2 = torch.acos(arg2_clamped)
        
        arg1 = (y_scaled * (l1_scaled + l2_scaled * torch.cos(theta2))  - x_scaled * l2_scaled * torch.sin(theta2))/(x_scaled * x_scaled + y_scaled * y_scaled+1)
        with torch.no_grad():
                arg1.data = torch.clamp(arg1.data, min=-0.999, max=0.999)
        arg1_clamped = F.hardtanh(arg1, -1, 1)
        theta1 = torch.asin(arg1_clamped)
        
        result[0,i,0] = theta1
        result[0,i,1] = theta2
    return result

def accurate(input, size):
    result = torch.zeros((1,size,2))
    for i in range(size):
        x = input[0,i,0]
        y = input[0,i,1]
        
        x_scaled = x
        y_scaled = y
        l1_scaled = l1
        l2_scaled = l2
        
        theta2 = torch.acos(((x_scaled*x_scaled)+(y_scaled*y_scaled)-(l1_scaled*l1_scaled)-(l2_scaled*l2_scaled))/(2*l1_scaled*l2_scaled))
        
        theta1 = torch.asin((y_scaled * (l1_scaled + l2_scaled * torch.cos(theta2)) - x_scaled * l2_scaled * torch.sin(theta2))/(x_scaled * x_scaled + y_scaled * y_scaled))
        result[0,i,0] = theta1
        result[0,i,1] = theta2
    return result
            