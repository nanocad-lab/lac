import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scipy.fft import dct

import ssim_torch_multi
import mul_wrapper

import time
import utils

BITSHIFT_FACTOR = True

def approx_division(value,divisor):
    if divisor == 0:
        out = value
    else:
        out = value/(divisor)
        approx_bitshift = torch.round(torch.log2(divisor))
        out.data = value/(2**approx_bitshift)
    return out

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

# Gate for selecting input
class BinarizeGate(nn.Module):
    def __init__(self, size=4):
        super(BinarizeGate, self).__init__()
        # The weight is initialized to select each input with equal probability
        self.register_parameter('weight', nn.Parameter(torch.ones(size)))
        # when fixed=True, the weight value is no longer updated and the input with highest weight is always selected
        self.fixed = False
        # size is the number of inputs to select from
        self.size = size
        self.constant_sel = None

    def forward(self, input, cost):
        # Convert weight into probability values
        # 
        weight_norm = F.softmax(self.weight, dim=-1)
        self.weight_norm = weight_norm
        select = None
        if self.fixed and self.last_sel != None:
            select = self.last_sel
        if self.fixed and self.constant_sel != None:
            select = self.constant_sel
        output, cost, mul_id =  BinarizeGrad.apply(input, cost, weight_norm, self.training, self.fixed, select)
        self.last_sel = mul_id
        #print(self.last_sel)
        # Sample an input from the probability values
        # sel_pred = torch.multinomial(weight_norm.data,1).reshape(-1)
        # weight_norm.data = torch.zeros_like(weight_norm.data)
        # weight_norm.data[sel_pred] = 1.
        return output, cost

class BinarizeGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cost, weight_norm, training, fixed, force_select):
        ctx.fixed = fixed
        if training:
            
            sel_pred = torch.multinomial(weight_norm.data,1).view(-1)
            
        else:
            sel_pred = torch.argmax(weight_norm.data,-1)
        if force_select != None:
            print(force_select)
            with torch.enable_grad():
                output = x[...,force_select]
            output_cost = cost[force_select]
        else:
            with torch.enable_grad():
                output = x[...,sel_pred.item()]
            output_cost = cost[sel_pred.item()]
        ctx.save_for_backward(x,output,cost)
        return output.data, output_cost.data, sel_pred.item()
    
    @staticmethod
    def backward(ctx, grad_output, grad_cost, mult_id):
        x, output, cost = ctx.saved_tensors
        # Gradient for the inputs
        if x.requires_grad:
            grad_x, = torch.autograd.grad(output, x, grad_output, only_inputs=True)
        else:
            grad_x = None
        # Gradient for the binary gate
        binary_grads = (grad_output.unsqueeze(-1)*x).sum(tuple(range(len(grad_output.size()))))
        # Gradient for the binary gate (cost)
        binary_grads_cost = (grad_cost.unsqueeze(-1)*cost)#.sum(tuple(range(len(grad_cost.size()))))
        # print(binary_grads.mean(), binary_grads_cost.mean())
        binary_grads = binary_grads + binary_grads_cost
        if ctx.fixed:
            return grad_x, None, None, None, None, None
        else:
            return grad_x, None, binary_grads, None, None, None
class PSNR:
    def __init__(self):
        self.name="PSNR"
    @staticmethod
    def __call__(img1, img2, max=255):
        psnrsum = 0.
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

def kernel_multiplier_approx(tker1, x, scale, mul_approx_index, sum=True, one_pixel=np.array([None])):
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
    in2 = torch.stack([in1 for _ in range(x.size(0))], dim=0)

    if one_pixel.size != 1 or one_pixel[0]!=None:
        full_resized = torch.zeros(output.size())
        for pixel in one_pixel:
            cut_output = output[:,pixel:pixel+1]
            cut_in2 = in2[:,pixel:pixel+1]
            cut_out = mul_wrapper.mul_wrapper(cut_output.reshape(1,cut_output.numel()), cut_in2.reshape(1,cut_output.numel()), mul_approx_index)
            cut_resized = cut_out.view_as(cut_output)
            full_resized[:,pixel:pixel+1] = cut_resized
        resized = full_resized
        #print("ONE PIXEL",full_resized.size())
        
    else:
        #multiply in parallel
        #Accurate version: out = in2.view(1,kernel_elems*amount)*output.view(1,kernel_elems*amount)
        #Approximate version: out = mult_approx.mult_approx(output.view(1,output.numel()), in2.view(1,output.numel()), mul_approx_index)
        out = mul_wrapper.mul_wrapper(output.view(1,output.numel()), in2.view(1,output.numel()), mul_approx_index)
        
        #fold the output of multiplication
        # resized = out.view(x.shape[1],kernel_elems, int(out.numel()/(kernel_elems*x.shape[1])))
        resized = out.view_as(output)
    
    if sum:
        #shape the output
        summed = torch.sum(resized,1).view_as(x)
        return summed
    else:
        return resized

def accurate_mult_only(x, input, size=3):
    input_uf = F.unfold(input, 3, padding=1, stride=1).permute(0,2,1)
    x_uf = x.reshape(-1)
    return (input_uf * x_uf).permute(0,2,1)

def sobel_post(gx, gy, size):
    g = torch.abs(gx) + torch.abs(gy)
    gmin = torch.min(g.view(size,-1),dim=1)[0]
    gmax = torch.max(g.view(size,-1),dim=1)[0]
    dx = (gmax-gmin).clamp(1,None).view(-1,1,1,1)
    result = (g-gmin.view(-1,1,1,1))/dx*255
    return result

def accurate(x, input, size, image_size=32, post=True):
    input = input[:size]
    x_trans = torch.transpose(x,2,3)
    if post:
        gx = F.conv2d(input.to(torch.float), weight=x.to(torch.float), padding=(1,1), groups=input.size(1))
        gy = F.conv2d(input.to(torch.float), weight=x_trans.to(torch.float), padding=(1,1), groups=input.size(1))

        g = torch.abs(gx) + torch.abs(gy)
        gmin = torch.min(g.view(size,-1),dim=1)[0]
        gmax = torch.max(g.view(size,-1),dim=1)[0]
        dx = (gmax-gmin).clamp(1,None).view(-1,1,1,1)
        result = (g-gmin.view(-1,1,1,1))/dx*255
        return result
    else:
        gx = accurate_mult_only(x.to(torch.float32), input.to(torch.float32))
        gy = accurate_mult_only(x_trans.to(torch.float32), input.to(torch.float32))
        return torch.stack([gx, gy], dim=0)

def approx(x, input, size, mult_approx_index, post=True):
    input = input[:size]
    j = mult_approx_index
    x = torch.reshape(x, (3,3))
    x_trans = torch.transpose(x,0,1)
    
    if post:
        gx = kernel_multiplier_approx(x, input.data, 1.0, mult_approx_index)
        gy = kernel_multiplier_approx(x_trans, input.data, 1.0, mult_approx_index)
        g = torch.abs(gx) + torch.abs(gy)
        gmin = torch.min(g.view(size,-1),dim=1)[0]
        gmax = torch.max(g.view(size,-1),dim=1)[0]
        dx = (gmax-gmin).clamp(1,None)
        result = torch.floor((g-gmin.view(-1,1,1,1))/dx.view(-1,1,1,1)*255)
        return result
    else:
        gx = kernel_multiplier_approx(x, input.data, 1.0, mult_approx_index, sum=False).float()
        gy = kernel_multiplier_approx(x_trans, input.data, 1.0, mult_approx_index, sum=False).float()
        return torch.stack([gx, gy], dim=0)

class Sobel_Forward_Model(nn.Module):
    
    # Define additional parameters needed
    def __init__(self, size=1000, dtype=torch.float32, mul_approx_func_arr=[1], nas=False, nas_area=1.0):
        super(Sobel_Forward_Model, self).__init__()
        
        # Define parameters to train
        self.register_parameter('weight', nn.Parameter(torch.from_numpy(np.tile(np.array([1., 0., -1., 2., 0., -2., 1., 0., -1.])/2,(len(mul_approx_func_arr),1)))))
        self.register_parameter('scale', nn.Parameter(torch.from_numpy(np.tile(np.array([1.])*1,(len(mul_approx_func_arr),1)))))
        # self.register_parameter('weight_factor', nn.Parameter(torch.from_numpy(np.tile(np.array([1.])*2,(len(mul_approx_func_arr),1)))))
        self.register_parameter('weight_factor', nn.Parameter(torch.from_numpy(np.array([1.])*2)))
        
        # Define constants
        self.register_buffer('weight_orig', nn.Parameter(torch.from_numpy(np.array([1., 0., -1., 2., 0., -2., 1., 0., -1.]))))
        self.dtype = dtype
        self.size = size
        self.mult_list = mul_approx_func_arr
        self.metric = SSIM()
        self.isNormalized = True
        self.nas = nas
        self.nas_area = nas_area

        if nas:
            cost_table = torch.zeros(len(mul_approx_func_arr)).to(torch.float32)
            for i,mul_code in enumerate(mul_approx_func_arr):
                cost_table[i] = utils.multiplier_area(mul_code)
            self.register_buffer("cost_table", cost_table)

        # Define binary gates
        # This is for the parallel version, so 9 gates for the nine multipliers
        if nas:
            for i in range(9):
                self.register_module("bin_gate_{0}".format(i), BinarizeGate(len(self.mult_list)))
        
        # Save calculated scale
        self.saved_scale = np.tile(np.array([1.]),(len(mul_approx_func_arr),1))
        
    # Define how approximate computation is done
    def forward(self, input, acc=False, image_size=32):
        if acc:
            pass
            #Never executes
        else:
            image_size=input.size(-1)
        #Calculate output for all multipliers
            # output = torch.zeros([len(self.mult_list), self.size,1,image_size,image_size], dtype=self.dtype)
            output = []
            # ii = 0
            for ii,j in enumerate(self.mult_list):
                weight_clamped = F.hardtanh(self.weight, -1, 1)
                weight_factor_clamped = F.hardtanh(self.weight_factor, 1, 255)
                weight_rounded = weight_clamped[ii]*weight_factor_clamped#[ii]
                
                weight_in = weight_rounded.view(1,1,3,3)
                
                if self.nas:
                    x_data = approx(weight_rounded, input, self.size, j, post=False)
                    x = accurate(weight_in, input, self.size, image_size=image_size, post=False)
                    pass
                else:
                    # Approxiamte convoluiton
                    x_data = approx(weight_rounded, input, self.size, j)
                    
                    # Accurate convolution (for gradients)
                    x = accurate(weight_in, input, self.size, image_size=image_size)
                # Plug in gradients
                x.data = x_data
                x = x/self.scale[ii].data
                output.append(x)
            # print(output[0].mean(), output[1].mean(), self.weight_factor)
            output = torch.stack(output, 0)
            # print(output.size())
            # time.sleep(100)
            if self.nas:
                output_pos_gate = []
                cost_pos_gate = []
                for i in range(9):
                    input_gate = output[:,:,:,i].permute(1,2,3,0).contiguous()
                    output_gate, cost_gate = getattr(self, "bin_gate_{0}".format(i))(input_gate, self.cost_table)
                    output_pos_gate.append(output_gate)
                    cost_pos_gate.append(cost_gate)
                output_pos = torch.stack(output_pos_gate, 0).sum(0)
                gx = output_pos[0].view_as(input[:self.size])
                gy = output_pos[1].view_as(input[:self.size])
                output_pos = sobel_post(gx, gy, self.size)
                cost_pos = torch.stack(cost_pos_gate, 0).mean(0)
                output = (output_pos, cost_pos)
        return output

    # Define how accurate computation should be done
    def target(self, input):
        weight_in = self.weight_orig.view(1,1,3,3)
        image_size = input.size(-1)
        x = accurate(weight_in, input, self.size, image_size=image_size)
        #x = nn.functional.conv2d(input.to(self.dtype), weight=Scale_Grad.apply(weight_in.to(self.dtype), lr_scale[j]), padding=(1,1), groups=input.size(1))  
        return x
    
class Gaussian_Forward_Model(nn.Module):
    # Define additional parameters needed
    def __init__(self, size=1000, dtype=torch.float32, mul_approx_func_arr=[1], nas=False, nas_area=1.0):
        super(Gaussian_Forward_Model, self).__init__()
        # Define parameters to train
        self.register_parameter('weight', nn.Parameter(torch.from_numpy(np.tile(np.array([1., 2., 1., 2., 4., 2., 1., 2., 1.]).astype(np.float32)/16,(len(mul_approx_func_arr),1)))))
        self.register_buffer('scale', torch.from_numpy(np.tile(np.array([32]).astype(np.float32)*1,(len(mul_approx_func_arr),1))))
        self.register_parameter('weight_factor', nn.Parameter(torch.from_numpy(np.tile(np.array([16.]).astype(np.float32),(len(mul_approx_func_arr),1)))))
        
        # Define constants
        self.register_buffer('weight_orig', nn.Parameter(torch.from_numpy(np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]]))))

        self.dtype = dtype
        self.size = size
        self.mult_list = mul_approx_func_arr
        self.metric = SSIM()
        self.isNormalized = False
        self.nas = nas
        self.nas_area = nas_area
        self.config = np.array([None])

        if nas:
            cost_table = torch.zeros(len(mul_approx_func_arr)).to(torch.float32)
            for i,mul_code in enumerate(mul_approx_func_arr):
                cost_table[i] = utils.multiplier_area(mul_code)
            self.register_buffer("cost_table", cost_table)

        # Define binary gates
        # This is for the parallel version, so 9 gates for the nine multipliers
        if nas:
            for i in range(9):
                self.register_module("bin_gate_{0}".format(i), BinarizeGate(len(self.mult_list)))
        
        #Save calculated scale
        self.saved_scale = np.tile(np.array([16.]),(len(mul_approx_func_arr),1))
        
    # Define how approximate computation is done
    def forward(self, input, acc=False, image_size=32):
        if acc:
            
            #Never executes
            x = kernel_multiplier_approx(self.weight.to(self.dtype), input.to(self.dtype), self.scale.data)
        else:

        #Calculate output for all multipliers separately
            output = []
            for ii,j in enumerate(self.mult_list):
                
                with torch.no_grad():
                    self.weight.data = torch.clamp(self.weight.data, min=.001, max=0.999)
                    self.weight_factor.data = torch.clamp(self.weight_factor.data, min=1.001, max=254.999)

                weight_clamped = F.hardtanh(self.weight, 0, 1)
                weight_factor_clamped = F.hardtanh(self.weight_factor, 1, 255)
                # Return a rounded version without rounding the master weight
                weight_rounded = QuantizeGrad.apply(weight_clamped[ii]*weight_factor_clamped[ii])
                weight_in = weight_rounded.view(1,1,3,3)

                #print(getattr(self, "bin_gate_{0}".format(ii)))
                if self.config[0] == None:
                    # Approximate convolution
                    x_data_pre_scale = kernel_multiplier_approx(weight_rounded, input.data, 1., j, sum=False)
                else:
                    one_pixel = np.squeeze(np.argwhere(self.config == ii), axis=1)
                    print(one_pixel)
                    x_data_pre_scale = kernel_multiplier_approx(weight_rounded, input.data, 1., j, sum=False, one_pixel=one_pixel)
                
                # if self.nas:
                x = accurate_mult_only(weight_rounded, input)
                
                
                # print(x_data_pre_scale.size(), x.size())
                if not self.nas:
                    x = x.sum(1).view_as(input)
                    x_data_pre_scale = x_data_pre_scale.sum(1).view_as(input)
                
                # Apply scale and plug gradients
                x_data = x_data_pre_scale#/self.scale[ii] #scale for data
                x.data = x_data.to(x.dtype)
                
                # Internalized scale calculation
                #if BITSHIFT_FACTOR:
                #    x = approx_division(x,self.scale[ii].data)#torch.max(x_data)/255.)
                #else:
                #    x = x/self.scale[ii].data

                output.append(x)
            output = torch.stack(output, 0)

            if self.nas:
                output_pos_gate = []
                cost_pos_gate = []
                for i in range(9):
                    input_gate = output[:,:,i].permute(1,2,0).contiguous()
                    output_gate, cost_gate = getattr(self, "bin_gate_{0}".format(i))(input_gate, self.cost_table)
                    #getattr(self, "bin_gate_{0}".format(i)).last_gate = 
                    #approx_division(output_gate,torch.max(output_gate)/255.)
                    
                    output_pos_gate.append(output_gate)
                    cost_pos_gate.append(cost_gate)
                output_pos = torch.stack(output_pos_gate, 0).sum(0).view_as(input)
                cost_pos = torch.stack(cost_pos_gate, 0).mean(0)
                if self.config[0] != None:
                    output_pos = output.sum(2).sum(0).view_as(input)
                    cost_pos = cost_pos.data
                print()
                output_pos = approx_division(output_pos,torch.max(output_pos)/255.)+0.00001
                output = (output_pos, cost_pos)
            

        return output

    # Define how accurate computation should be done
    def target(self, input):
        input = torch.reshape(input,(1,self.size,32,32))
        weight_in = torch.reshape(torch.stack([self.weight_orig.to(self.dtype) for _ in range(self.size)]), (self.size, 1, 3, 3))
        output1 = nn.functional.conv2d(input.to(self.dtype), weight=weight_in, padding=(1,1), groups=input.size(1))
        return output1

def accurate_sharp(x, input, size):
    target1 = nn.functional.conv2d(input.to(torch.float), weight=x.to(torch.float), padding=(1,1), groups=input.size(1))

    size = input.size(0)
    with torch.no_grad():
        output_max = target1.view(size,-1).max(1)[0].view(size,1,1,1)
        output_min = target1.view(size,-1).min(1)[0].view(size,1,1,1)
    mod = ((target1-output_min)/((output_max-output_min).clamp(1e-6,None)))*255.
    # print("Accurate", x.min(), x.max(), input.size())
    sharpened = input + mod
    with torch.no_grad():
        sharpened_max = sharpened.view(size,-1).max(1)[0].view(size,1,1,1)
        sharpened_min = sharpened.view(size,-1).min(1)[0].view(size,1,1,1)
    result = torch.clamp(((sharpened-sharpened_min)/(sharpened_max-sharpened_min).clamp(1e-6,255))*255, 0, 255)
    result.data = torch.floor(result.data)
    # result = torch.floor(torch.clamp(((sharpened-sharpened_min)/(sharpened_max-sharpened_min).clamp(1e-6,255))*255, 0, 255))
    # print("Accurate", result.mean().item())
    return result

def approx_sharp(x, input, size, mult_approx_index):
    j = mult_approx_index
    conv = kernel_multiplier_approx(x, input.data, 1., j)

    size = input.size(0)
    output_max = conv.view(size,-1).max(1)[0].view(size,1,1,1)
    output_min = conv.view(size,-1).min(1)[0].view(size,1,1,1)
    mod = ((conv-output_min)/((output_max-output_min).clamp(1e-6,None)))*255.
    # print("Approx", x.min(), x.max(), input.size())
    sharpened = input + mod
    sharpened_max = sharpened.view(size,-1).max(1)[0].view(size,1,1,1)
    sharpened_min = sharpened.view(size,-1).min(1)[0].view(size,1,1,1)
    result = torch.floor(torch.clamp(((sharpened-sharpened_min)/(sharpened_max-sharpened_min).clamp(1e-6,255))*255, 0, 255))
    # print("Approx", result.mean().item())
    return result

class Laplacian_Forward_Model(nn.Module):
    
    # Define additional parameters needed
    def __init__(self, size=1000, dtype=torch.float32, mul_approx_func_arr=[1], nas=False):
        super(Laplacian_Forward_Model, self).__init__()
        
        # Define parameters to train
        self.register_parameter('weight', nn.Parameter(torch.from_numpy(np.tile(np.array([-1., -1., -1., -1., 8., -1., -1., -1., -1.])/8,(len(mul_approx_func_arr),1)))))
        self.register_parameter('scale', nn.Parameter(torch.from_numpy(np.tile(np.array([1.])*1,(len(mul_approx_func_arr),1)))))
        self.register_parameter('weight_factor', nn.Parameter(torch.from_numpy(np.tile(np.array([1.])*8,(len(mul_approx_func_arr),1)))))
        
        # Define constants
        self.register_buffer('weight_orig', nn.Parameter(torch.from_numpy(np.array([-1., -1., -1., -1., 8., -1., -1., -1., -1.]))))
        self.dtype = dtype
        self.size = size
        self.mult_list = mul_approx_func_arr
        self.metric = SSIM()
        self.isNormalized = True
        
        # Save calculated scale
        self.saved_scale = np.tile(np.array([1.]),(len(mul_approx_func_arr),1))
        
    # Define how approximate computation is done
    def forward(self, input, acc=False, image_size=32):
        if acc:
            #Never executes
            x = kernel_multiplier_approx(self.weight.to(self.dtype), input.to(self.dtype), self.scale.data)
        else:

        #Calculate output for all multipliers
            output = []
            for ii,j in enumerate(self.mult_list):
                                
                # Clamping
                with torch.no_grad():
                    self.weight.data = torch.clamp(self.weight.data, min=-0.999, max=0.999)
                    self.weight_factor.data = torch.clamp(self.weight_factor.data, min=1.001, max=254.999)
#                 weight_in = torch.reshape(torch.stack([torch.clamp(self.weight[j].to(self.dtype), min=0, max=256) for _ in range(self.size)]), (self.size, 1, 3, 3))
                # Clamping with gradient. Might be the same as normal clamp
                
                weight_clamped = F.hardtanh(self.weight, -1, 1)
                weight_factor_clamped = F.hardtanh(self.weight_factor, 1, 255)
                
                weight_rounded = weight_clamped[ii]*weight_factor_clamped[ii]

                weight_in = weight_rounded.view(1,1,3,3)
                
                # Approxiamte convoluiton
                
                x_data = approx_sharp(weight_rounded, input, self.size, j)
                # Apply scale for gradients
                #scale_x = torch.sum(torch.abs(weight_in))
                #if scale_x == 0:
                #    scale_x = 0.0000001
                #mod_weight = weight_in/scale_x #scale for gradients
                
                # Accurate convolution (for gradients)
                
                x = accurate_sharp(weight_in, input, self.size)
                
                # Apply scale to data
                #scale = (torch.mean(x_data_pre_scale)/torch.mean(self.target(input)))
                #self.saved_scale[j] = scale.data.numpy()
                #x_data = x_data_pre_scale/1.0 #scale #scale for data
                
                # Plug in gradients
                x.data = x_data
                output.append(x)
            output = torch.stack(output, 0)
        return output

    # Define how accurate computation should be done
    def target(self, input):
        weight_in = self.weight_orig.view(1,1,3,3)

        x = accurate_sharp(weight_in, input, self.size)
        #x = nn.functional.conv2d(input.to(self.dtype), weight=Scale_Grad.apply(weight_in.to(self.dtype), lr_scale[j]), padding=(1,1), groups=input.size(1))  
        return x
    

class DCT_Forward_Model(nn.Module):
    
    # Define additional parameters needed
    def __init__(self, size=1000, dtype=torch.float32, mul_approx_func_arr=[1]):
        super(DCT_Forward_Model, self).__init__()
        # Define parameters to train
        dct_mat = dct(np.eye(8), axis=0, norm='ortho')
        weigth_one_multiplier = np.reshape(dct_mat, (1, 64))
        
        self.register_parameter('weight', nn.Parameter(torch.from_numpy(np.tile(weigth_one_multiplier,(len(mul_approx_func_arr),1)))))
        
        self.register_parameter('scale', nn.Parameter(torch.from_numpy(np.tile(np.array([1.]).astype(np.float64)*1,(len(mul_approx_func_arr),1)))))
        self.register_parameter('weight_factor', nn.Parameter(torch.from_numpy(np.tile(np.array([65000.]).astype(np.float64),(len(mul_approx_func_arr),1)))))
        
        # Define constants
        self.register_buffer('weight_orig', nn.Parameter(torch.from_numpy(weigth_one_multiplier)))
        self.dtype = dtype
        self.size = size
        self.mult_list = mul_approx_func_arr
        self.metric = PSNR()
        self.isNormalized = False
        
        #Save calculated scale
        self.saved_scale = np.tile(np.array([16.]),(len(mul_approx_func_arr),1))
        
    # Define how approximate computation is done
    def forward(self, input):

        #Calculate output for all multipliers separately
            # output = torch.zeros([len(self.mult_list), 1, self.size,32,32], dtype=self.dtype)
            output = []
            for ii,j in enumerate(self.mult_list):
                # Return a rounded version without rounding the master weight
                weight_rounded =  self.weight[ii]#*self.weight_factor[ii] #QuantizeGrad.apply(weight_clamped[ii]*weight_factor_clamped[ii])

                x = grad_dct(weight_rounded, input, self.weight_factor[ii], self.size)

                x_data = approx_dct(torch.unsqueeze(weight_rounded, 0), input, self.weight_factor[ii], self.size, j)

                x.data = x_data
                output.append(x)
            output = torch.stack(output, 0)
            return output

    # Define how accurate computation should be done
    def target(self, input):
        target = accurate_dct(self.weight_orig, input, self.size)
        return target

def slice_smart(image, size=8):
    # image = image.view(image.size(0),1,image.size(2),image.size(3))
    im_slice = torch.nn.functional.unfold(image, size, stride=size)
    im_slice = im_slice.permute(0,2,1).reshape(-1,size,size)
    return im_slice

def fold_smart(im_slice, size=8):
    im_slice = im_slice.view(-1,16,size*size).permute(0,2,1)
    image = F.fold(im_slice, 32, size, stride=size)
    return image

# def approx_dct(x, input, scale, size, mult_approx_index):
#     input = input[:size]
#     j = mult_approx_index
#     kernel_size = 512
        
#     #scale = 65535
#     x_initial = x #torch.Tensor(np.ones((8,8))) #self.weight
#     x_adjusted = torch.reshape(x_initial, (8,8))*scale
#     x_adjusted_transpose = torch.transpose(x_adjusted, 0, 1)

#     # Perform DCT
#     result = torch.zeros((1,size,32,32))
#     image = input.data-128

#     im_slice = slice_smart(image, 8)

#     temp11 = approx_matmul(x_adjusted, im_slice, 8, j, swap_inputs=True)/scale
#     temp22 = approx_matmul(temp11, x_adjusted_transpose, 8, j, swap_inputs=True)/scale
#     temp = torch.round(temp22/q_50+0.000001)
                
#     temp23 = torch.reshape(mul_wrapper.mul_wrapper(torch.tile(torch.reshape(q_50, (1, 64))[0],(1,16*size)), torch.reshape(temp, (1, 64*16*size))[0], j), (16*size, 8,8))
                
#     temp = approx_matmul(x_adjusted_transpose, temp23, 8, j, swap_inputs=True)/scale
#     temp2 = approx_matmul(temp, x_adjusted, 8, j, swap_inputs=True)/scale
        
#     temp_out = torch.round(temp2)+128
#     result = fold_smart(temp_out).view(1,-1,32,32)
                
#     #print(result)
#     return result

# def accurate_dct(x, input, size):
#     input = input[:size]
#     scale = 65535
#     x_initial = x #torch.Tensor(np.ones((8,8))) #self.weight
#     x_adjusted = torch.reshape(x_initial, (8,8)).to(torch.float)
#     x_adjusted_transpose = torch.transpose(x_adjusted, 0, 1)
    
#     result = torch.zeros((1,size,32,32))
#     image = input.data-128

#     im_slice = slice_smart(image, 8)

#     # Forward DCT
#     temp11 = torch.matmul(x_adjusted, im_slice.to(torch.float))
#     temp22 = torch.matmul(temp11, x_adjusted_transpose)
#     temp = torch.round(temp22/q_50+0.000001)
                
#     temp23 = torch.reshape(torch.mul(torch.tile(torch.reshape(q_50, (1, 64))[0],(1,16*size)), torch.reshape(temp, (1, 64*16*size))[0]), (16*size, 8,8))
                
#     temp = torch.matmul(x_adjusted_transpose, temp23)
#     temp2 = torch.matmul(temp, x_adjusted)
        
#     temp_out = torch.round(temp2)+128
#     result = fold_smart(temp_out).view(1,-1,32,32)
#     return result            
                            
# def grad_dct(x, input, scale, size):
#     input = input[:size]
#     #scale = 65535
#     x_initial = QuantizeGrad.apply(x*scale) #torch.Tensor(np.ones((8,8))) #self.weight
#     x_adjusted = torch.reshape(x_initial, (8,8)).to(torch.float)
#     x_adjusted_transpose = torch.transpose(x_adjusted, 0, 1)
    
#     result = torch.zeros((1,size,32,32))
#     image = input.data-128
    
#     im_slice = slice_smart(image, 8)    

#     # Forward DCT
#     temp11 = QuantizeGrad.apply(torch.matmul(x_adjusted, im_slice))/scale
#     temp22 = QuantizeGrad.apply(torch.matmul(temp11, x_adjusted_transpose))/scale
                
#     # Pointwise
#     temp = torch.round(temp22/q_50+0.000001)
#     temp23 = QuantizeGrad.apply(torch.reshape(torch.mul(torch.tile(torch.reshape(q_50, (1, 64))[0],(1,16*size)), torch.reshape(temp, (1, 64*16*size))[0]), (16*size, 8,8)))

#     # Inverse DCT
#     temp = QuantizeGrad.apply(torch.matmul(x_adjusted_transpose, temp23))/scale
#     temp2 = QuantizeGrad.apply(torch.matmul(temp, x_adjusted))/scale
        
#     temp_out = temp2+128
#     result = fold_smart(temp_out).view(1,-1,32,32)
#     return result

def approx_dct(x, input, scale, size, mult_approx_index):
                j = mult_approx_index
                kernel_size = 512
                    
                #scale = 65535
                x_initial = x #torch.Tensor(np.ones((8,8))) #self.weight
                x_adjusted = torch.reshape(x_initial, (8,8))*scale
                x_adjusted_transpose = torch.transpose(x_adjusted, 0, 1)
                
                debug = 0
                # Perform DCT
                result = torch.zeros((1,size,32,32))
                image = input[:,0].data-128
                
                
                    
                im_slice = torch.zeros((16*size,8,8))
                
                for i in range(size):     
                    for hindex in [1,2,3,4]:
                        for vindex in [1,2,3,4]:
                            im_slice[i*16+(hindex-1)*4+vindex-1] = image[i][(hindex-1)*8:hindex*8,(vindex-1)*8:vindex*8]
                            
                temp11 = torch.zeros((16*size,8,8))

                temp11 = approx_matmul(x_adjusted, im_slice, 8, j, swap_inputs=True)/scale
                    
                    
                    
                temp22 = torch.zeros((16*size,8,8))
                            
                temp22 = approx_matmul(temp11, x_adjusted_transpose, 8, j, swap_inputs=True)/scale
                          
                        
                temp23 = torch.zeros((16*size,8,8))
                    
                temp = torch.zeros((16*size,8,8))
                
                for i in range(size):   
                    for hindex in [1,2,3,4]:
                        for vindex in [1,2,3,4]:
                            temp[i*16+(hindex-1)*4+vindex-1] = torch.round(torch.div(temp22[i*16+(hindex-1)*4+vindex-1], q_50)+0.000001)
                            
                temp23 = torch.reshape(mul_wrapper.mul_wrapper(torch.tile(torch.reshape(q_50, (1, 64))[0],(1,16*size)), torch.reshape(temp, (1, 64*16*size))[0], j), (16*size, 8,8))
                            
                temp = approx_matmul(x_adjusted_transpose, temp23, 8, j, swap_inputs=True)/scale
                temp2 = approx_matmul(temp, x_adjusted, 8, j, swap_inputs=True)/scale
                    
                temp_out = torch.round(temp2)+128
                for i in range(size):     
                    for hindex in [1,2,3,4]:
                        for vindex in [1,2,3,4]:    
                            result[0,i,(hindex-1)*8:hindex*8,(vindex-1)*8:vindex*8] = temp_out[i*16+(hindex-1)*4+vindex-1]
                            
                #print(result)
                return result

def accurate_dct(x, input, size):
                j = 0
                scale = 65535
                x_initial = x #torch.Tensor(np.ones((8,8))) #self.weight
                x_adjusted = torch.reshape(x_initial, (8,8)).to(torch.float)
                x_adjusted_transpose = torch.transpose(x_adjusted, 0, 1)
                
                result = torch.zeros((1,size,32,32))
                image = input[:,0].data-128
                
                im_slice = torch.zeros((16*size,8,8)).to(torch.double)
                
                for i in range(size):     
                    for hindex in [1,2,3,4]:
                        for vindex in [1,2,3,4]:
                            im_slice[i*16+(hindex-1)*4+vindex-1] = image[i][(hindex-1)*8:hindex*8,(vindex-1)*8:vindex*8]
                            

                temp11 = torch.matmul(x_adjusted, im_slice.to(torch.float))
                            
                temp22 = torch.matmul(temp11, x_adjusted_transpose)
                          
                        
                temp23 = torch.zeros((16*size,8,8))
                    
                temp = torch.zeros((16*size,8,8))
                
                for i in range(size):   
                    for hindex in [1,2,3,4]:
                        for vindex in [1,2,3,4]:
                            temp[i*16+(hindex-1)*4+vindex-1] = torch.round(torch.div(temp22[i*16+(hindex-1)*4+vindex-1], q_50)+0.000001)
                            
                temp23 = torch.reshape(torch.mul(torch.tile(torch.reshape(q_50, (1, 64))[0],(1,16*size)), torch.reshape(temp, (1, 64*16*size))[0]), (16*size, 8,8))
                            
                temp = torch.matmul(x_adjusted_transpose, temp23)
                temp2 = torch.matmul(temp, x_adjusted)
                    
                temp_out = torch.round(temp2)+128
                for i in range(size):     
                    for hindex in [1,2,3,4]:
                        for vindex in [1,2,3,4]:    
                            result[0,i,(hindex-1)*8:hindex*8,(vindex-1)*8:vindex*8] = temp_out[i*16+(hindex-1)*4+vindex-1]
                return result            
                            
def grad_dct(x, input, scale, size):
                j = 0
                #scale = 65535
                x_initial = QuantizeGrad.apply(x*scale) #torch.Tensor(np.ones((8,8))) #self.weight
                x_adjusted = torch.reshape(x_initial, (8,8)).to(torch.float)
                x_adjusted_transpose = torch.transpose(x_adjusted, 0, 1)
                
                result = torch.zeros((1,size,32,32))
                image = input[:,0].data-128
                
                im_slice = torch.zeros((16*size,8,8)).to(torch.double)
                
                for i in range(size):     
                    for hindex in [1,2,3,4]:
                        for vindex in [1,2,3,4]:
                            im_slice[i*16+(hindex-1)*4+vindex-1] = image[i][(hindex-1)*8:hindex*8,(vindex-1)*8:vindex*8]
                            

                temp11 = QuantizeGrad.apply(torch.matmul(x_adjusted, im_slice.to(torch.float)))/scale
                            
                temp22 = QuantizeGrad.apply(torch.matmul(temp11.to(torch.double), x_adjusted_transpose.to(torch.double)))/scale
                          
                        
                temp23 = torch.zeros((16*size,8,8))
                    
                temp = torch.zeros((16*size,8,8))
                
                for i in range(size):   
                    for hindex in [1,2,3,4]:
                        for vindex in [1,2,3,4]:
                            temp[i*16+(hindex-1)*4+vindex-1] = torch.round(torch.div(temp22[i*16+(hindex-1)*4+vindex-1], q_50)+0.000001)
                            
                temp23 = QuantizeGrad.apply(torch.reshape(torch.mul(torch.tile(torch.reshape(q_50, (1, 64))[0],(1,16*size)), torch.reshape(temp, (1, 64*16*size))[0]), (16*size, 8,8)))
                            
                temp = QuantizeGrad.apply(torch.matmul(x_adjusted_transpose, temp23))/scale
                temp2 = QuantizeGrad.apply(torch.matmul(temp, x_adjusted.to(torch.double)))/scale
                    
                temp_out = temp2+128
                for i in range(size):     
                    for hindex in [1,2,3,4]:
                        for vindex in [1,2,3,4]:    
                            result[0,i,(hindex-1)*8:hindex*8,(vindex-1)*8:vindex*8] = temp_out[i*16+(hindex-1)*4+vindex-1]
                return result

            
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