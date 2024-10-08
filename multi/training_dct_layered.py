#!/usr/bin/env python
# coding: utf-8

# ### Load relevant modules

# In[2]:
PRE_SELECTION = True

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

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
sys.path.insert(0, '../applications')
sys.path.insert(0, '../utils')

import torch.nn.functional as F


# In[4]:


#import applications
import model_gauss_factor
import ssim_torch_multi

# ### Additional Imports

# In[5]:


from scipy.linalg import dft

# In[7]:
from PIL import Image

# Read the image
image_preview = None #Image.open('../images/cameraman.tif')

# Define a transform to convert the image to tensor
transform = transforms.ToTensor()

# Convert the image to PyTorch tensor
tensor_preview = None #transform(image_preview)*255.

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

# class BinarizeGate(nn.Module):
#     def __init__(self, size=4):
#         super(BinarizeGate, self).__init__()
#         # The weight is initialized to select each input with equal probability
#         self.register_parameter('weight', nn.Parameter(torch.ones(size)))
#         # when fixed=True, the weight value is no longer updated and the input with highest weight is always selected
#         self.fixed = False
#         # size is the number of inputs to select from
#         self.size = size
#     def forward(self, input, pre_sels=None):
#         # Convert weight into probability values
#         # 
#         weight_norm = F.softmax(self.weight, dim=-1)
#         self.weight_norm = weight_norm
#         output =  BinarizeGrad.apply(input, weight_norm, self.training, self.fixed)
#         # Sample an input from the probability values
#         sel_pred = torch.multinomial(weight_norm.data,1).reshape(-1)
#         weight_norm.data = torch.zeros_like(weight_norm.data)
#         weight_norm.data[sel_pred] = 1.
#         return output

# class BinarizeGrad(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, weight_norm, training, fixed):
#         ctx.fixed = fixed
#         if training:
#             sel_pred = torch.multinomial(weight_norm.data,1).view(-1)
#         else:
#             sel_pred = torch.argmax(weight_norm.data,-1)
#         with torch.enable_grad():
#             output = x[...,sel_pred.item()]

#         global s_sel
        
#         s_sel =  sel_pred

#         #print(sel_pred)
#         #save to global variables for graphing
        
        
#         ctx.save_for_backward(x,output)
#         return output.data
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         x, output = ctx.saved_tensors

#         #Gradient for kernels (same as Backward pass for Sum)
        
#         if x.requires_grad:
#             grad_x, = torch.autograd.grad(output, x, grad_output, only_inputs=True)#torch.tile(grad_output,(1,len(model_saved.mult_list)))
#         else:
#             grad_x = None
        
#         # Gradient for the binary gate
#         #print(grad_output.size())
#         #print(torch.transpose(grad_output,2,3).size())
#         #x = torch.transpose(x,2,3)
#         #grad_output = torch.transpose(grad_output,2,3)
#         binary_grads = (grad_output.unsqueeze(-1)*x).sum(tuple(range(len(grad_output.size()))))
        
#         #print("X",x)
#         #print("OUT",grad_output)
#         #print("BIN",binary_grads)

#         if ctx.fixed:
#             return grad_x, None, None, None, None
#         else:
#             return grad_x, binary_grads, None, None, None


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
    def forward(self, input, total_loss=None, pre_sels=None):
        # Convert weight into probability values
        # 
        weight_norm = F.softmax(self.weight, dim=-1)
        self.weight_norm = weight_norm
        print("INPUT ", input.size())
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
            sel_count = 1
            if pre_sels is None:
                sel_pred = torch.multinomial(weight_norm.data,sel_count).view(-1)
            else:
                sel_pred = pre_sels
        else:
            sel_pred = torch.argmax(weight_norm.data,-1)
        with torch.enable_grad():
            out_stacked = []
            out_stacked.append(x[...,sel_pred.item()])
            full_out = torch.cat(out_stacked)

        global s_sel
        
        s_sel =  sel_pred
      
        ctx.save_for_backward(x,full_out,total_loss)
        return full_out.data
    
    @staticmethod
    def backward(ctx, grad_output):
        x, output, total_loss = ctx.saved_tensors

        #Gradient for kernels (same as Backward pass for Sum)
        if x.requires_grad:
            if PRE_SELECTION:
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
def print_testing_psnr(file=False, text_file="", verbal=True):
    input = testX_gray[:,0:0+20]
        
    # Calculate target (correct output)
    layer1.model.size = 20
    layer2.model.size = 20
    layer3.model.size = 20

    layer1.fixed_multiplier = selected_saved[0].data
    layer2.fixed_multiplier = selected_saved[1].data
    layer3.fixed_multiplier = selected_saved[2].data
    
    
    out_layer1 = layer1.step(i,input)[0]
    selected_layer1 = s_sel.data[0]
    print("Iter ",i," Selected layer 1: ",layer1.fixed_multiplier)#s_sel.data[0]

    out_layer2 = layer2.step(i,out_layer1)[0]
    selected_layer2 = s_sel.data[0]
    print("Selected layer 2: ",layer2.fixed_multiplier)

    out_layer3 = layer3.step(i,out_layer2)[0]
    selected_layer3 = s_sel.data[0]
    print("Selected layer 3: ",layer3.fixed_multiplier)

    final_target = layer3.model.target(input).clone().data
    loss = -layer3.criterion(out_layer3.unsqueeze(0), final_target , 255)
        
    # Calculate loss for each multiplier
    
    
    if verbal:
        if file:
            print("Testing loss:", file=text_file)
            print("Sum of PSNR: {0}".format(int(loss*1000)/1000), file=text_file)
        else:
            print("Testing loss:")
            print("PSNR: {0}".format(int(loss*1000)/1000))
    return (layer1.fixed_multiplier, layer2.fixed_multiplier, layer3.fixed_multiplier, int(loss*1000)/1000)
        
def save_preview_image(model,name="../images/preview_image_latest.png",mult=0):
    input = tensor_preview.unsqueeze(0)
    
    # Calculate target (correct output)
    model.size = 1
    
    # Calculate output
    output = model(input, acc=False, image_size=512)
    target = model.target(input).clone().data
    
    if not model.isNormalized:
        for j in range(len(model_saved.mult_list)):
            output[j] = output[j]/model.scale[j]
        
    torchvision.utils.save_image(output[mult,0,0].to(torch.float)/255,name)
    torchvision.utils.save_image(target.to(torch.float)/255,"../images/target.png")


#Define PSNR loss



# ### Define forward pass
class NAS_layer:
    def __init__(self, input_size, optimizer, scheduler, model, acc=False, train=False, size=1000, verbal=False):
        self.input_size = input_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.train = train
        self.size = size
        self.verbal = verbal
        self.fixed_multiplier = -1

        global model_saved
        model_saved = model
        
        #Initialize bingate optimizer
        
        self.gate = BinarizeGate(size=len(model.mult_list))
        self.optimizer_bingate = torch.optim.SGD(self.gate.parameters(), lr=0.0001)
        
        # Enable training
        self.gate.train()
        
        # Define loss function
        self.criterion = model.metric
        
        #training set location
        train_set_loc = 0
        
        # Checkpoints
        self.checkpoint_steps = 20
        self.last_criterions = np.ones((len(model.mult_list),self.checkpoint_steps + 1))*1000
        self.model_dicts = np.array([model.state_dict()]*(self.checkpoint_steps + 1))
        self.model_factors = np.array([model.state_dict()]*(self.checkpoint_steps + 1))
        #initialize arrays for loss for each multiplier
        self.loss_arr = [1000]*len(model.mult_list)
        
        #Global variables for graphs
        global iii
        iii = 0
        global s_lr
        global s_sum 
        global s_psnr
        global s_sel
        s_lr = [0]*5000
        
        self.first_input = trainX_gray[:,train_set_loc:train_set_loc+input_size]

    def step(self, i, input, pre_train_stage=False):

        if pre_train_stage:
            output = []
            for m in range(input.size(0)):
                output.append(self.model(input[m], multipliers_list = [m]))
            output_all = torch.cat(tuple(output),dim=0)
            return output_all
        else:#TODO

            if self.fixed_multiplier == -1:

                pre_sels = None
                if PRE_SELECTION:
                    weight_norm = F.softmax(self.gate.weight, dim=-1)
                    pre_sels = torch.multinomial(weight_norm,1).view(-1)

                    output = self.model(input, multipliers_list = [pre_sels.item()])#, multipliers_list = pre_sels)
                    #print("NEW ", output.size())
                    ext_size = list(output.size())
                    ext_size[0] = 11
                    output_ext = torch.zeros(ext_size, dtype=torch.float32)
                    output_ext[pre_sels] = output[0]
                    #print("THIS ", output_ext.size())
                    # output_s = self.gate(torch.permute(output,(1,2,3,4,0)).contiguous())
                    permuted = torch.permute(output_ext,(1,2,3,4,0)).contiguous()
                    output_s = self.gate(permuted, None, pre_sels=pre_sels)
            else:
                output = self.model(input)
                output_s = output[self.fixed_multiplier]

            return output_s
    
    def step2(self, i,penalty=0):
        
        if self.train:
            self.optimizer_bingate.step()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.optimizer_bingate.zero_grad()
        

def forward(input_size, optimizer, scheduler, model, models=[], acc=False, train=False, size=1000, verbal=False, timed=False, enable_checkpoints=False, randomize=False, threshold=100):
    '''
    optimizer, approximate model, accurate model
    acc=True assumes that func_app is differentiable; acc=False assumes that it's not
    train=False means in training mode, =True means in validation mode
    size=number of data points tested
    timed=True will output time taken for each iteration
    enable_checkpoints=True will save model state and then load the best one of {checkpoint_steps} iterations
    rndomize=True will randomize the dataset if no improvement for {checkpoint steps} iterations
    
    '''

    area_arr = torch.Tensor([1.01, 0.74, 0.03, 0.07, 0.13, 0.07, 0.21, 0.14, 0.5, 0.25, 0.39])
    #area_arr = torch.Tensor([1.01, 0.74, 0.07, 0.13, 0.07, 0.21, 0.25, 0.39])
    layer1_best_weight = None
    
    global selected_saved
    selected_saved = []
    global s_sel
    global layer1
    global layer2
    global layer3

    
    layer1 = NAS_layer(input_size, optimizer, scheduler, models[0], acc, train, size, verbal)
    layer2 = NAS_layer(input_size, optimizer, scheduler, models[1], acc, train, size, verbal)
    layer3 = NAS_layer(input_size, optimizer, scheduler, models[2], acc, train, size, verbal)
    input = trainX_gray[:,0:0+input_size]

    input_pre_train = torch.stack([input for _ in range(len(models[0].mult_list))])

    for i in range(10):
        pre_out_layer1 = layer1.step(0,input_pre_train, pre_train_stage=True)
        pre_out_layer2 = layer2.step(0,pre_out_layer1, pre_train_stage=True)
        pre_out_layer3_list = []
        for m in range(pre_out_layer2.size(0)):
            temp_l3 = layer3.step(0,pre_out_layer2[m], pre_train_stage=True)
            pre_out_layer3_list.append(temp_l3)
        pre_out_layer3 = torch.cat(tuple(pre_out_layer3_list),dim=0)

        final_target = layer3.model.target(layer3.first_input).clone().data
        final_loss = 0
        for m in range(pre_out_layer3.size()[0]):
            loss_m = layer3.criterion(pre_out_layer3[m].unsqueeze(0), final_target , 255)
            final_loss -= loss_m
            #print(loss_m)
        print("PSNR",final_loss)
        final_loss.backward()

        layer1.step2(0)
        layer2.step2(0)
        layer3.step2(0)

    best_criterion = 10000
    for i in range(size):
        
        out_layer1 = layer1.step(i,input)[0]
        selected_layer1 = s_sel.data[0]
        print("Iter ",i," Selected layer 1: ",s_sel.data[0])#s_sel.data[0]

        out_layer2 = layer2.step(i,out_layer1)[0]
        selected_layer2 = s_sel.data[0]
        print("Selected layer 2: ",s_sel.data[0])

        out_layer3 = layer3.step(i,out_layer2)[0]
        selected_layer3 = s_sel.data[0]
        print("Selected layer 3: ",s_sel.data[0])

        
        #             self.model.weight[jj] = self.model_dicts[np.argmin(self.last_criterions[jj])][jj]
        #             self.model.weight_factor[jj] = self.model_factors[np.argmin(self.last_criterions[jj])][jj]
        #     

        L1 = torch.sum(layer1.gate.weight*area_arr)#/100# layer1.gate.weight[selected_layer_1] #
        #L1.data = area_arr[selected_layer1] #(area_arr[selected_layer_1]+area_arr[selected_layer_2]+area_arr[selected_layer_2])
        L2 = torch.sum(layer2.gate.weight*area_arr)#/100
        #L2.data = area_arr[selected_layer2]
        L3 = torch.sum(layer3.gate.weight*area_arr)#/100
        #L3.data = area_arr[selected_layer3]
        total_area = area_arr[selected_layer1]+area_arr[selected_layer2] +area_arr[selected_layer3]

        if (total_area) > threshold:
            penalty = ((L1+L2+L3)-threshold)*300
        else:
            penalty = 0

        final_target = layer3.model.target(layer3.first_input).clone().data
        final_loss = -layer3.criterion(out_layer3.unsqueeze(0), final_target , 255)
        print("PSNR",final_loss)
        #print("PSNR with penalty",(final_loss+penalty))
        (final_loss+penalty).backward()

        if (best_criterion > final_loss) and ((total_area) <= threshold):
            with torch.no_grad():
                layer1_best_weight = layer1.model.weight.clone()
                layer1_best_factor = layer1.model.weight_factor.clone()
                layer1_best_bingate = layer1.gate.weight.clone()
                layer2_best_weight = layer2.model.weight.clone()
                layer2_best_factor = layer2.model.weight_factor.clone()
                layer2_best_bingate = layer1.gate.weight.clone()
                layer3_best_weight = layer3.model.weight.clone()
                layer3_best_factor = layer3.model.weight_factor.clone()
                layer3_best_bingate = layer1.gate.weight.clone()
                best_criterion = final_loss
                selected_saved = [selected_layer1,selected_layer2,selected_layer3]
        print("Best so far: ", best_criterion)
        print("Best multipliers so far: ", selected_saved)
                

        #print(final_target.grad)

        #print(layer3.gate.weight.grad)
        
        layer1.step2(i,100*L1)
        layer2.step2(i,100*L2)
        layer3.step2(i,100*L3)

        if (layer1_best_weight != None and i!=0 and (i%20==0)) or i == size-1:
            with torch.no_grad():
                layer1.model.weight.data = layer1_best_weight
                layer1.model.weight_factor.data = layer1_best_factor
                layer1.gate.weight.data = layer1_best_bingate
                layer2.model.weight.data = layer2_best_weight
                layer2.model.weight_factor.data = layer2_best_factor
                layer2.gate.weight.data = layer2_best_bingate
                layer3.model.weight.data = layer3_best_weight
                layer3.model.weight_factor.data = layer3_best_factor
                layer3.gate.weight.data = layer3_best_bingate
                

        #with torch.no_grad():
        #        scale = torch.mean(out_layer3)/torch.mean(final_target)+0.000001
        #        out_layer3 = out_layer3/scale.data
        print(-layer3.criterion(out_layer3.unsqueeze(0), final_target , 255))
        
        print("Layer 1 bingate:",layer1.gate.weight)
        print("Layer 2 bingate:",layer2.gate.weight)
        print("Layer 3 bingate:",layer3.gate.weight)
    return best_criterion


# ### Prepare optimizer and train. Change the input size to match your application.



loss_pre = 0
