#!/usr/bin/env python
# coding: utf-8

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
import time

torch.manual_seed(0)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
sys.path.insert(0, '../applications')
sys.path.insert(0, '../utils')

import torch.nn.functional as F


# In[19]:


import ssim_torch_multi
import model_dct_factor as model_dct_factor
import model_dct_layer1_bestgrad as model_dct_layer1
import model_dct_layer2_bestgrad as model_dct_layer2
import model_dct_layer3_bestgrad as model_dct_layer3
import training_dct_layered as training
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


loss_pre = 0
THRESHOLD = 0.2

#lr_vals - array of learning rates, iters - number of iterations, size - number of images in the model, verbal - get some additional information
def perform_train(lr_vals, iters=40, size = 10, verbal=False):
    global loss_pre
    global output_weights
    global model_saved
    scores = np.array(lr_vals)
    for i in range(len(lr_vals)):
        dtype = torch.float32
        input_size = size
        model = model_dct_layer1.Forward_Model(size,mul_approx_func_arr=mul_approx_func_arr)
        model2 = model_dct_layer2.Forward_Model(size,mul_approx_func_arr=mul_approx_func_arr)
        model3 = model_dct_layer3.Forward_Model(size,mul_approx_func_arr=mul_approx_func_arr)
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr_vals[i], momentum=0.9)
        optimizer = torch.optim.Adam([
                {'params': model.weight},
                {'params': model.weight_factor, 'lr': 0, 'gamma':0.3}
            ], lr=lr_vals[i])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.95, step_size=10, verbose=False)
         # Specifies the annealing strategy
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.90, step_size=10, verbose=True)
        #loss_pre = training.forward(input_size, optimizer, scheduler, model, models=[model], train=False, size=1)
        #print("Accuracy before training is: {0} , Learning rate: {1}".format(loss_pre, lr_vals[i]))
        scores[i] = training.forward(input_size, optimizer, scheduler, model, models=[model,model2,model3], train=True, size=iters,
                            verbal=verbal, timed=False, enable_checkpoints = False, threshold=THRESHOLD) 
    return scores


#List of multipliers to train
#mul_approx_func_arr = [1, 2, 5, 12, 7, 6, 11, 10]
mul_approx_func_arr = [1, 2, 0, 5, 12, 7, 6, 3, 4, 11, 10]
area_arr = [1.01, 0.74, 0.03, 0.07, 0.13, 0.07, 0.21, 0.14, 0.5, 0.25, 0.39]

#0 - mul8u_JV3, 1 - mul16s_GK2, 2 - mul16s_GAT, 3 - EMT, 4 - EMT_16, 5 - mul8u_FTA, 6 - modded1KVL,
#7 - mul8s_1KR3, 8 - kulkarni_16bit, 9 - kulkarni_8bit, 10 - DRUM_16bit, 11 - DRUM_16bit_4
#set for gauss: [0, 1, 2, 3, 4, 5]
#optimize kernel

max_psnr = 127*8*8

#0.0666
#0.107388
#multi:
#0.0666
#0.085977

#torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
#lr_vals - array of learning rates, iters - number of iterations, 
#size - number of images in the model, verbal - get some additional information
for t in [0.85, 0.83, 0.8, 0.77, 0.75, 0.7, 0.6, 0.5, 0.93, 1.0, 1.01]:
    THRESHOLD = t
    
    print("Training set (Before):")
    model = model_dct_layer1.Forward_Model(100,mul_approx_func_arr=mul_approx_func_arr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr_vals[i], momentum=0.9)
    optimizer = torch.optim.Adam([
    {'params': model.weight},
    {'params': model.weight_factor, 'lr': 0, 'factor':0.1}
    ], lr=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)
    #loss_pre = training.forward(100, optimizer, scheduler, model,models=[model] train=False, size=1, verbal=True)
    print("Testing set (Before):")
    #training.print_testing_psnr(verbal=True)
    print()

    start = time.time()

    #0.01
    perform_train([0.001], iters=1301, size=100, verbal=False) 
    print()

    print("Bingate weights")
    print("Layer 1 bingate:",training.layer1.gate.weight)
    print("Layer 2 bingate:",training.layer2.gate.weight)
    print("Layer 3 bingate:",training.layer3.gate.weight)
    print()

    print("Training set (After):")
    #loss_pre = training.forward(100, optimizer, scheduler, training.model_saved, train=False, size=1, verbal=True)
    print()

    print("Testing set (After):")
    mult1, mult2, mult3, test_val = training.print_testing_psnr(verbal=False)

    enablePrint()
    f = open("multi.out", "a")
    f.write(str(area_arr[mult1] + area_arr[mult2] + area_arr[mult3]) +  " " + str(test_val) + "\n")
    f.close()
    blockPrint()
    #print("time: ",time.time() - start)