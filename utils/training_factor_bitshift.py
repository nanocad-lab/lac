#!/usr/bin/env python
# coding: utf-8

# ### Load relevant modules

# In[2]:


import torch
import torch.nn as nn
import numpy as np
import math
import cv2
from torchmetrics import StructuralSimilarityIndexMeasure
import copy
import time
import random
import torchvision
import torchvision.transforms as transforms
import sys
import model_sharp_unscaled
import model_dft_fast
import model_in2k
import model_dct_factor

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
np.set_printoptions(precision=3)
sys.path.insert(0, '../applications')
sys.path.insert(0, '../utils')

import torch.nn.functional as F

CEILING = False
BITSHIFT = True
NON_IMAGE = False
LIGHT_NAS = False
LIGHT_NAS_SELS = 1
PRE_SELECTION = True
BEST_METER = False
area_arr = torch.Tensor([1.01, 0.74, 0.03, 0.07, 0.13, 0.07, 0.21, 0.14, 0.5, 0.25, 0.39])
# In[4]:

#computes approximate division (based on bitshift) result and correct gradients
def approx_division(value,divisor,ceil=False):
    out = value/(divisor)
    global CEILING
    if CEILING:
        approx_bitshift = torch.ceil(torch.log2(divisor))
        #print("SHARP",2**approx_bitshift)
    else:
        approx_bitshift = torch.round(torch.log2(divisor))
    out.data = value/(2**approx_bitshift)
    return out

def get_bitshift_divisor(value,ceil=False):
    global CEILING
    if CEILING:
        divisor = 2**(torch.ceil(torch.log2(torch.max(value)/255.)))
        #print("SHARP",divisor)
    else:
        divisor = 2**(torch.round(torch.log2(torch.max(value)/255.)))
    if torch.isnan(divisor) or divisor==0:
        divisor = 1
    return divisor



#import applications
import model_gauss_factor
import ssim_torch_multi

# ### Additional Imports

# In[5]:


from scipy.linalg import dft

# In[7]:
from PIL import Image

# Read the image
image_preview = Image.open('../images/cameraman.tif')

# Define a transform to convert the image to tensor
transform = transforms.ToTensor()

# Convert the image to PyTorch tensor
tensor_preview = transform(image_preview)*255.


trainset = torchvision.datasets.CIFAR10(root='../cifar_data', train=True, download=True, transform=transforms.ToTensor())     
testset = torchvision.datasets.CIFAR10(root='../cifar_data', train=False, download=True, transform=transforms.ToTensor())    
rgb2g = torch.tensor([0.2989, 0.587, 0.114])

trainX_gray = torch.zeros((1,1000,32,32))
testX_gray = torch.zeros((1,1000,32,32))

for i in range(1000):
    trainX_gray[0,i] = torch.round((trainset[i][0][0]*rgb2g[0]+trainset[i][0][1]*rgb2g[1]+trainset[i][0][2]*rgb2g[2])*255)
for i in range(1000):
    testX_gray[0,i] = torch.round((testset[i][0][0]*rgb2g[0]+testset[i][0][1]*rgb2g[1]+testset[i][0][2]*rgb2g[2])*255)

#global variables for graphing
global s_lr
global s_sum 
global s_psnr
global s_sel
s_lr = [0]*5000
s_sum  = [0]*5000
s_psnr = [0]*5000
s_sel = [0]*5000


# ### Define approximate computations

# ### Define kernel and multipliers

# In[9]:


gaussian_kernel = np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]])


# ### Binarized gate

def get_bingate_weight():
    return gate.weight

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
    def forward(self, input, total_loss, pre_sels=None):
        # Convert weight into probability values
        # 
        weight_norm = F.softmax(self.weight, dim=-1)
        self.weight_norm = weight_norm
        output =  BinarizeGrad.apply(input, weight_norm, self.training, self.fixed, total_loss, pre_sels)
        # Sample an input from the probability values
        sel_pred = torch.multinomial(weight_norm.data,1).reshape(-1)
        weight_norm.data = torch.zeros_like(weight_norm.data)
        weight_norm.data[sel_pred] = 1.
        return output

class BinarizeGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_norm, training, fixed, total_loss, pre_sels):
        ctx.fixed = fixed
        if training:
            sel_count = LIGHT_NAS_SELS
            if pre_sels is None:
                sels = torch.multinomial(weight_norm.data,sel_count).view(-1)
            else:
                sels = pre_sels
            sel_pred = sels[0]
        else:
            sel_pred = torch.argmax(weight_norm.data,-1)
        with torch.enable_grad():
            out_stacked = []
            for sel in range(sel_count):
                out_stacked.append(x[...,sels[sel].item()])
            full_out = torch.cat(out_stacked)
        
        #save to global variables for graphing
        global s_sel
        global iii
        s_sel[iii] =  sel_pred
        iii+=1
        
        ctx.save_for_backward(x,full_out,total_loss)
        return full_out.data
    
    @staticmethod
    def backward(ctx, grad_output):
        x, output, total_loss = ctx.saved_tensors

        #Gradient for kernels (same as Backward pass for Sum)
        if x.requires_grad:
            if LIGHT_NAS:
                grad_x, = torch.autograd.grad(output, x, grad_output, only_inputs=True)
            else:
                #grad_x, = torch.autograd.grad(output, x, grad_output, only_inputs=True)
                grad_x = torch.tile(grad_output,(1,len(model_saved.mult_list)))
        else:
            grad_x = None
        
        # Gradient for the binary gate
        binary_grads = (grad_output.unsqueeze(-1)*x).sum(tuple(range(len(grad_output.size()))))
        
        if ctx.fixed:
            return grad_x, None, None, None, None, None
        else:
            return grad_x, binary_grads, None, None, None, None


# ### Forward model class

# ### Auxillary functions

# In[44]:


#print testing psnr for each multiplier separately and weight
def print_testing_psnr(file=False, text_file="", verbal=True, test_size=20):
    print(NON_IMAGE)
    if NON_IMAGE:
        input = torch.reshape(testX_gray[0,0:200],(1,200,2))
        model_saved.size = 200
    else:
        test_set_loc = 0
        input = testX_gray[:,test_set_loc:test_set_loc+test_size]
        model_saved.size = test_size
        
    # Calculate target (correct output)
    
    
    target = model_saved.target(input).clone().data
    
    if not model_saved.isNormalized:
            target = approx_division(target,torch.max(target)/255.)+0.00001

    # Calculate output
    if CEILING:
        output = model_saved(input, use_saved_divisor2=True)#bug fix from edits
    else:
        output = model_saved(input)
    
    print("Divisor_1_used: ",end='')
    if not model_saved.isNormalized:
        for j in range(len(model_saved.mult_list )):
            divisor = get_bitshift_divisor(output[j])
            output[j] = output[j]/divisor
        #print(divisor.item(),end=" ")
        #print(torch.isnan(output).any())
    print()
    print("Divisor_2_used (scale): ",end='')    
    #if model_saved.weight_factor != None:
    #    print(np.reshape(2**torch.round(torch.log2(model_saved.weight_factor)).data.numpy(),(1,-1))[0])

    #for im in diff[0,0]:
        #print(torch.mean(abs(im)))
        #print(torch.max(abs(im)))

    criterion = model_saved.metric
        
    # Calculate loss for each multiplier
    
    loss = 0
    loss_arr = [1000]*len(model_saved.mult_list)
    for j in range(len(model_saved.mult_list)): 
        loss1 = -criterion(output[j], target, 255)
        loss_arr[j] = int(float(loss1.data)*1000)/1000
        loss += loss1
    if verbal:
        if file:
            print("Testing_loss:", file=text_file)
            print(loss_arr, file=text_file)
            print("Sum of PSNR: {0}".format(int(loss*100)/100), file=text_file)
        else:
            print("Testing_loss: ", end='')
            print(loss_arr)
            print("Sum of PSNR: {0}".format(int(loss*100)/100))
    return loss_arr
        
def save_preview_image(model,name="../images/preview_image_latest.png",mult=0):
    input = tensor_preview.unsqueeze(0)
    
    # Calculate target (correct output)
    model.size = 1
    
    # Calculate output
    output = model(input, acc=False, image_size=512)
    target = model.target(input).clone().data
    
    if not model_saved.isNormalized:
            target = approx_division(target,torch.max(target)/255.)+0.00001

    if not model_saved.isNormalized:
        for j in range(len(model_saved.mult_list)):
            # scale_temp = (torch.max(output[j])/255.)
            # if torch.isnan(scale_temp) or scale_temp<=0:
            #     scale_temp = 1

            divisor = get_bitshift_divisor(model_saved.scale[j].data)
            output[j] = output[j]/divisor
        
    torchvision.utils.save_image(output[mult,0,0].to(torch.float)/255,name)
    torchvision.utils.save_image(target.to(torch.float)/255,"../images/target.png")


#Define PSNR loss



# ### Define forward pass

# In[38]:


def forward(input_size, optimizer, scheduler, model, train=False, size=1000, verbal=False, verbose_level=2, light_nas=False, light_nas_sels=1, nas_lr=0.1, single_mode=False, target_acc=False, target_acc_val=0.0, perf_con=False, perf_con_value=100., HYPER_DELTA=100.0, HYPER_GAMMA=0.9):
    '''
    optimizer, approximate model, accurate model
    acc=True assumes that func_app is differentiable; acc=False assumes that it's not
    train=False means in training mode, =True means in validation mode
    size=number of data points tested
    timed=True will output time taken for each iteration
    enable_checkpoints=True will save model state and then load the best one of {checkpoint_steps} iterations
    rndomize=True will randomize the dataset if no improvement for {checkpoint steps} iterations
    
    '''
    global model_saved
    global CEILING
    global NON_IMAGE
    global LIGHT_NAS
    global LIGHT_NAS_SELS
    LIGHT_NAS_SELS = light_nas_sels
    LIGHT_NAS = light_nas
    NON_IMAGE = type(model) == model_in2k.Forward_Model
    COMPLEX_WEIGHT = type(model)==model_dft_fast.Forward_Model
    CEILING = type(model)==model_sharp_unscaled.Forward_Model
    USE_FACTOR = type(model) != model_in2k.Forward_Model
    BEST_METER = type(model)== model_dct_factor.Forward_Model or type(model)== model_gauss_factor.Forward_Model or type(model)== model_dft_fast.Forward_Model
    model_saved = model

    if NON_IMAGE:
        trainX_gray_non_image = torch.load('../axbench_data/in2k/train_tensor.pt')
        testX_gray_non_image = torch.load('../axbench_data/in2k/test_tensor.pt')
        trainX_gray.data = trainX_gray_non_image.data
        testX_gray.data = testX_gray_non_image.data
    
    #Initialize bingate optimizer
    global gate
    gate = BinarizeGate(size=len(model.mult_list))
    if light_nas:
        optimizer_bingate = torch.optim.Adam(gate.parameters(), lr=nas_lr)
    else:
        optimizer_bingate = torch.optim.SGD(gate.parameters(), lr=2)
    
    # Enable training
    if(train):
        gate.train()
    
    # Define loss function
    if target_acc:
        criterion = model.limited_metric
    else:
        criterion = model.metric
    
    #training set location
    train_set_loc = 0#temp
    
    # Checkpoints
    checkpoint_steps = 20
    if BEST_METER:
        best_criterions = np.ones(len(model.mult_list))
        best_weights = [0]*len(model.mult_list)
        best_factors = [0]*len(model.mult_list)
        best_weights_imag = [0]*len(model.mult_list)
        best_weights_real = [0]*len(model.mult_list)

    last_criterions = np.ones((len(model.mult_list),checkpoint_steps + 1))*1000
    model_dicts = np.array([model.state_dict()]*(checkpoint_steps + 1))
    model_factors = np.array([model.state_dict()]*(checkpoint_steps + 1))
    model_dicts_real = np.array([model.state_dict()]*(checkpoint_steps + 1))
    model_dicts_imag = np.array([model.state_dict()]*(checkpoint_steps + 1))
    #initialize arrays for loss for each multiplier
    loss_arr = [1000]*len(model.mult_list)
    
    #Global variables for graphs
    global iii
    iii = 0
    global s_lr
    global s_sum 
    global s_psnr
    global s_sel
    global area_arr
    s_lr = [0]*5000

    if target_acc:
        weight_norm = 1
    
    
    for i in range(size):
        
        # Generate input
        if NON_IMAGE:
            input = torch.reshape(trainX_gray[0,train_set_loc:train_set_loc+input_size], (1, input_size, 2))
        else:
            input = trainX_gray[:,train_set_loc:train_set_loc+input_size]
        
        
        # Calculate target (correct output)
        target = model.target(input).clone().data

        pre_sels = None
        if PRE_SELECTION and light_nas:
            weight_norm = F.softmax(gate.weight, dim=-1)
            pre_sels = torch.multinomial(weight_norm,LIGHT_NAS_SELS).view(-1)
            if perf_con:
                print(pre_sels[0])
                area_sel = area_arr[pre_sels[0]]

        # Calculate output
        output = model(input, mults=pre_sels)
        

        #if train:
        if not model.isNormalized:
            target = approx_division(target,torch.max(target)/255.)+0.00001
        
            for j in range(len(model.mult_list)):
                with torch.no_grad():
                    model.scale[j] = torch.max(output[j])/255.
                    if torch.isnan(model.scale[j]) or model.scale[j]<=0:
                        model.scale[j] = 1
                    #print(model.scale[j].data)
                output[j] = approx_division(output[j],model.scale[j].data)+0.00001
                #print("aft: {0}, mean: {1}, max: {2}".format(torch.min(output[j]), torch.mean(output[j]), torch.max(output[j])))
        
        #else:
         #   for j in range(len(mul_approx_func_arr)):
          #      output[j] = output[j]/model.scale[j]
                    #print(model.scale[j])
        #print(output)         
        
        # Calculate loss for each multiplier
        loss = 0
        loss_arr_tens = [1000]*len(model.mult_list)
        loss_arr = [1000]*len(model.mult_list)
        for j,m in enumerate(model.mult_list): 
            if target_acc:
                loss1 = -criterion(output[j], target, 255, binw=weight_norm, target_acc_val=target_acc_val, mult=j)#mults=torch.tensor(model.mult_list, dtype=torch.int)[pre_sels])
            else:
                if perf_con: #other possible option is area_arr[j]
                    loss1 = -criterion(output[j], target, 255) + HYPER_DELTA * F.relu(torch.Tensor([area_arr[j] - HYPER_GAMMA*perf_con_value])[0])# + torch.Tensor([100])[0]
                #     if j == pre_sels[0]:
                #         loss1 = -criterion(output[j], target, 255) + HYPER_DELTA * F.relu(torch.Tensor([area_sel - HYPER_GAMMA*perf_con_value])[0]) #PERF_CON_LOSS
                #     else:
                #         loss1 = torch.Tensor([0])[0]#-criterion(output[j], target, 255) + 100 #HYPER_DELTA * F.relu(torch.Tensor([area_arr[j] - HYPER_GAMMA*perf_con_value])[0])
                else:
                    loss1 = -criterion(output[j], target, 255)
            #print((torch.mean(output[j])/torch.mean(target)))
            loss_arr[j] = float(loss1.data)
            loss_arr_tens[j] = loss1
            loss += loss1

        if light_nas:
            if NON_IMAGE:
                permuted = torch.permute(output,(1,2,3,0)).contiguous()
            else:
                permuted = torch.permute(output,(1,2,3,4,0)).contiguous()
            output_s = gate(permuted, None, pre_sels=pre_sels)

        with torch.no_grad():
            if COMPLEX_WEIGHT:
                model_dicts_real[i%checkpoint_steps] = model.weight_real.clone()
                model_dicts_imag[i%checkpoint_steps] = model.weight_imag.clone()
            else:
                model_dicts[i%checkpoint_steps] = model.weight.clone()
                if USE_FACTOR:
                    model_factors[i%checkpoint_steps] = model.weight_factor.clone()
            #print(model_dicts)
            for jj in range(len(model.mult_list)):
                last_criterions[jj,i%checkpoint_steps] = loss_arr[jj]

                if BEST_METER:
                    if loss_arr[jj] < best_criterions[jj]:
                        best_criterions[jj] = loss_arr[jj]
                        if COMPLEX_WEIGHT:
                            best_weights_real[jj] = model.weight_real[jj].clone()
                            best_weights_imag[jj] = model.weight_imag[jj].clone()
                        else:
                            best_weights[jj] = model.weight[jj].clone()
                            if USE_FACTOR:
                                best_factors[jj] = model.weight_factor[jj].clone()
                    
        
        #bingate optimizer step
        optimizer_bingate.zero_grad()
        stacked = torch.stack(loss_arr_tens,-1)
        

        if light_nas:
                loss_bingate = 0
                for j in range(output_s.size(0)):
                    if target_acc:
                        loss_bingate += -criterion(output_s[j].unsqueeze(0), target, 255, binw=weight_norm, target_acc_val=target_acc_val, mult=j)#, mults=torch.tensor(model.mult_list, dtype=torch.int)[pre_sels])
                    else:
                        if perf_con:
                            loss_bingate += -criterion(output_s[j].unsqueeze(0), target, 255)
                        else:
                            loss_bingate += -criterion(output_s[j].unsqueeze(0), target, 255)
        else:
            if single_mode:
                loss_bingate = torch.sum(stacked)
            else:
                output_s = gate(torch.unsqueeze(stacked, 0), torch.sum(stacked))
                loss_bingate = output_s[0]
        
        #save to global variables for grpahs
        s_sum[i] = sum(stacked.data.numpy())
        s_psnr[i] = stacked.data.numpy()

        if perf_con:
            loss_bingate = loss_bingate + HYPER_DELTA * F.relu(area_arr[j] - torch.Tensor([HYPER_GAMMA*perf_con_value])[0])
        
        if train:
            loss_bingate.backward()
            #print("Gradients: {0}\n".format(model.weight.grad))
            if (light_nas or perf_con) and not target_acc:
                optimizer_bingate.step()
            else:
                if (i!=0 and (i%checkpoint_steps)==0):
                    optimizer_bingate.step()
                
        
         
        #kernel optimizer step   
            #print(model.weight.grad)
            optimizer.step()
            
            scheduler.step()
            #scheduler.step(loss)
            
            optimizer.zero_grad()
        
        #print(model.weight_factor)
        if ((i%checkpoint_steps)==(checkpoint_steps-1)):
            if COMPLEX_WEIGHT:
                for jj in range(len(model.mult_list)):
                    with torch.no_grad():
                        model.weight_real[jj] = model_dicts_real[np.argmin(last_criterions[jj])][jj]
                        model.weight_imag[jj] = model_dicts_imag[np.argmin(last_criterions[jj])][jj]
                with torch.no_grad():
                    model_dicts_real[checkpoint_steps] = model.weight_real.clone()
                    model_dicts_imag[checkpoint_steps] = model.weight_imag.clone()
            else:
                for jj in range(len(model.mult_list)):
                    with torch.no_grad():
                        model.weight[jj] = model_dicts[np.argmin(last_criterions[jj])][jj]
                        if USE_FACTOR:
                            model.weight_factor[jj] = model_factors[np.argmin(last_criterions[jj])][jj]
                        if BEST_METER:
                            if COMPLEX_WEIGHT:
                                model.weight_real[jj] = best_weights_real[jj]
                                model.weight_imag[jj] = best_weights_imag[jj]
                            else:
                                model.weight[jj] = best_weights[jj]
                                if USE_FACTOR:
                                    model.weight_factor[jj] = best_factors[jj]
                        
                with torch.no_grad():
                    model_dicts[checkpoint_steps] = model.weight.clone()
                    if USE_FACTOR:
                        model_factors[checkpoint_steps] = model.weight_factor.clone()
            model_saved = copy.deepcopy(model)
            optimizer.zero_grad()
            
        #printing
        if i%1==0:
            if verbal:
                if verbose_level == 1:
                    print("Iter_{0}: {1}".format(i,stacked.data.numpy()))
                else:
                    print("Iter: ",i, " PSNR of all: ", stacked.data.numpy())
                    print("Loss of selected: {:.3f}\n".format(loss_bingate))
                    print(gate.weight)
                
                #print("Weights: {0}\n".format(model.weight.data.numpy()))
                #print("Gradients: {0}\n".format(model.weight.grad))
                
                #Another style
                #print("Iter: {0} , PSNR: {3}, \nPSNR in single mod: {4} , \nCurrent learning rates: {5} ,\nBest learing rates: {6}"
                 #     .format(i+1, model.weight.data.numpy().astype(np.uint8), model.scale.data.numpy(), np.around(loss_arr,decimals=2), psnr_single, np.around(lr_scale,decimals=2), lr_best))
    return stacked