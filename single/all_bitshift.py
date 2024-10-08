#!/usr/bin/env python
# coding: utf-8

# ### Load relevant modules

# In[2]:

import matplotlib.pyplot as plt
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
import os

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
sys.path.insert(0, '../applications')
sys.path.insert(0, '../utils')

import torch.nn.functional as F

# In[4]:

import training_factor_bitshift as training 
#import applications
import model_gauss_factor
import model_sobel_unscaled
import model_sharp_unscaled
import model_dft_fast
import model_dct_factor
import model_in2k
import ssim_torch_multi

trained_h_results_0 = [0.987, 0.822, 0.974, 0.98, 0.988, 0.17, 0.151, 0.976, 0.711, 0.957, 0.965]
trained_h_results_1 = [0.998, 0.696, 0.023, 0.002, 0.041, 0.22, 0.496, 0.972, 0.465, 0.901, 0.992]
trained_h_results_2 = [0.96, 0.898, 0.924, 0.96, 0.749, 0.749, 0.811, 0.004, 0.632, 0.985, 0.959]
trained_h_results_3 = [30.796, 29.575, 26.235, 26.16, 28.121, 27.632, 27.863, 28.944, 27.651, 31.107, 30.939]
trained_h_results_4 = [54.837, 45.273, 35.006, 41.744, 35.343, 35.006, 44.741, 40.76, 40.132, 57.28, 61.721]
trained_h_results_5 = [0.104, 0.831, 0.787, 0.514, 0.135, 1.649, 0.104, 0.273, 0.648, 0.282, 0.123]
# ### Additional Imports

global model_type
torch.manual_seed(0)
#0 - Gaussian
#1 - Sobel
#2 - Sharp
#3 - DFT
#4 - JPEG
#5 - IN2K
APP = 0
#Visual
COPY_FRIENDLY = True
CRITERION = "Area"
PREVIEW = False
INCLUDE_ACCURATE = False
VERBAL = True
#NAS settings
SINGLE = False
PERF_NAS = False
AREA_LIMIT = 100
LIGHT_NAS = True
LIGHT_NAS_SELS = 2
TARGET_ACC = False #Optimize hardware under accuracy constraint
TARGET_ACC_VAL = 0.94

#Size
MAX_ITERS = 201
train_set_size = 10
TEST_SIZE = 20
#Hyper parameters
LR = 1.
NAS_LR = 0.1
FACTOR_DECAY_RATE = 0.1
WEIGHT_DECAY_RATE = 0.95

#PERF_CON MODE
PERF_CON = False
PERF_CON_VALUE = 0.1
if PERF_CON:
    TARGET_ACC = False
    SINGLE = False
    PERF_NAS = False
    LIGHT_NAS = False
    LIGHT_NAS_SELS = 1


def main():
    mode = 0
    if (mode == 0):
        single_conf()
    if (mode == 1):
        if TARGET_ACC:
            vary_acc()
        else:
            vary_thres()
        

def vary_acc():
    global mul_approx_func_arr
    global area_arr
    global TARGET_ACC
    global TARGET_ACC_VAL
    TARGET_ACC = True
    LIGHT_NAS = True
    SINGLE = False
    PERF_NAS = False
    AREA_LIMIT = 100
    res_area = []
    res_acc = []
    for acc in [0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]:
        TARGET_ACC_VAL = acc
        blockPrint()
        train_before, test_before, train_after, test_after, selected = run1()
        enablePrint()
        area_sel = area_arr[initial_mul_approx_func_arr.index(mul_approx_func_arr[selected])]
        res_area.append(area_sel)
        res_acc.append(abs(test_after[selected]))
        print(TARGET_ACC_VAL," ",area_arr[selected], " ", abs(test_after[selected]))
    # plt.title(PLOT_NAME + " (Accuracy constraint)")
    # plt.plot(res_area, res_acc, color = 'r', label="Accuracy constraint")
    # global trained_h_results
    # plt.plot(area_arr, trained_h_results, "o", label="Trained hardware")
    # plt.ylim(plot_lim)
    # plt.xlabel("Area")
    # plt.ylabel("SSIM")
    # plt.legend()
    # plt.show()


def vary_thres():
    global mul_approx_func_arr
    global area_arr
    global PERF_NAS
    global AREA_LIMIT
    global PERF_CON_VALUE
    PERF_NAS = True
    if CRITERION == "Area":
        iterate_thres = [100., 0.4, 0.3, 0.2, 0.1]
    if CRITERION == "Delay":
        iterate_thres = [100., 2.5, 2., 1.5, 1.]
    for thres in iterate_thres:
        PERF_CON_VALUE = thres
        AREA_LIMIT = thres
        blockPrint()
        train_before, test_before, train_after, test_after, selected = run1()
        enablePrint()
        area_sel = area_arr[initial_mul_approx_func_arr.index(mul_approx_func_arr[selected])]
        print(" ",area_sel, " ", abs(test_after[selected]))

def single_conf():
    start = time.time()
    BLOCK_PRINT = False
    if BLOCK_PRINT:
        blockPrint()
    train_before, test_before, train_after, test_after, selected = run1()
    if BLOCK_PRINT:
        enablePrint()

    print("Selected: ", abs(test_after[selected]))
    if TARGET_ACC:
        print("Area: ",area_arr[selected])
    print("Runtime: ", time.time()-start)

#inner
NO_FACTOR = False
if not LIGHT_NAS:
    LIGHT_NAS_SELS = 1

if TARGET_ACC:
    LIGHT_NAS = False
    SINGLE = False
    PERF_NAS = False
    AREA_LIMIT = 100

if CRITERION == "Area":
    area_arr = [1.01, 0.74, 0.03, 0.07, 0.13, 0.07, 0.21, 0.14, 0.5, 0.25, 0.39]
    #area_arr_indexed = [0.03, 1.01, 0.74, 0.14, 0.5, 0.07, 0.21, 0.07, 100, 100, 0.39, 0.25, 0.13]
    initial_mul_approx_func_arr = [1, 2, 0, 5, 12, 7, 6, 3, 4, 11, 10]
    #initial_mul_approx_func_arr = [12]
    #area_arr = [1]
if CRITERION == "Delay":
    area_arr = [2.95, 2.57, 0.58, 0.95, 1.41, 0.89, 1.33]
    initial_mul_approx_func_arr = [1, 2, 0, 5, 12, 7, 6]

mul_approx_func_arr = []

if APP == 0:
    model_type = model_gauss_factor
    LR = 1.
    NAS_LR = 0.01
    PLOT_NAME="Gaussian blur"
    trained_h_results = trained_h_results_0
    plot_lim = (0.9,1)
if APP == 1:
    model_type = model_sobel_unscaled
    LR = 0.003
    NAS_LR = 0.03
    PLOT_NAME="Sobel"
    trained_h_results = trained_h_results_1
    plot_lim = (0,1)
if APP == 2:
    model_type = model_sharp_unscaled
    LR = 0.02
    NAS_LR = 0.01
    PLOT_NAME="Laplacian"
    trained_h_results = trained_h_results_2
    plot_lim = (0.6,1)
if APP == 3:
    model_type = model_dft_fast
    LR = 0.1
    NO_FACTOR = True
    NAS_LR = 0.3
    PLOT_NAME="DFT"
    trained_h_results = trained_h_results_3
    plot_lim = (20,40)
if APP == 4:
    model_type = model_dct_factor
    LR = 0.03
    NAS_LR = 0.005 #0.01 good
    #FACTOR_DECAY_RATE = 0.3
    #WEIGHT_DECAY_RATE = 0.96
    PLOT_NAME="DCT"
    trained_h_results = trained_h_results_4
    plot_lim = (30,65)
if APP == 5:
    model_type = model_in2k
    LR = 4.
    NO_FACTOR = True
    WEIGHT_DECAY_RATE = 0.95
    train_set_size = 1000
    NAS_LR = 1.0
    PLOT_NAME="In2jk"
    trained_h_results = trained_h_results_5
    plot_lim = (0,1)




#lr_vals - array of learning rates, iters - number of iterations, size - number of images in the model, verbal - get some additional information
def perform_train(lr_vals, iters=40, size = 10, verbal=False, mul_approx_func_arr=[0]):
    global output_weights
    global model_saved
    for i in range(len(lr_vals)):
        acc = False
        dtype = torch.float32
        input_size = size
        model = model_type.Forward_Model(size,mul_approx_func_arr=mul_approx_func_arr)
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr_vals[i], momentum=0.9)
        if NO_FACTOR:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_vals[i])
        else:
            optimizer = torch.optim.Adam([
                    {'params': model.weight},
                    {'params': model.weight_factor, 'lr': 100, 'factor':FACTOR_DECAY_RATE}
                ], lr=lr_vals[i])
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=WEIGHT_DECAY_RATE, step_size=1)
        loss_pre = training.forward(input_size, optimizer, scheduler, model, train=False, size=1)

        loss_end = training.forward(input_size, optimizer, scheduler, model, train=True, size=iters, nas_lr=NAS_LR,
                            verbal=verbal, light_nas=LIGHT_NAS, light_nas_sels=LIGHT_NAS_SELS, single_mode=SINGLE, 
                            target_acc=TARGET_ACC, target_acc_val=TARGET_ACC_VAL, perf_con=PERF_CON, perf_con_value=PERF_CON_VALUE) 
    return loss_pre



# In[42]:
def run1():
    global mul_approx_func_arr
    global area_arr
    global initial_mul_approx_func_arr
    if PERF_NAS:
        mul_approx_func_arr = []

        for i in range(len(initial_mul_approx_func_arr)):
            if area_arr[i] <= AREA_LIMIT:
                mul_approx_func_arr.append(initial_mul_approx_func_arr[i])
    else:
        if INCLUDE_ACCURATE:
            mul_approx_func_arr = [1, 2, 0, 5, 12, 7, 6, 3, 4, 11, 10, -1]
        else:
            mul_approx_func_arr = initial_mul_approx_func_arr

    print("Training set (Before):")
    model = model_type.Forward_Model(train_set_size,mul_approx_func_arr=mul_approx_func_arr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr_vals[i], momentum=0.9)
    if NO_FACTOR:
        optimizer = torch.optim.Adam(model.parameters(), lr=0)
    else:
        optimizer = torch.optim.Adam([
            {'params': model.weight},
            {'params': model.weight_factor, 'lr': 0, 'factor':0.1}
        ], lr=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)
    train_before = training.forward(train_set_size, optimizer, scheduler, model, train=False, size=1, verbal=True)
    print("Testing set (Before):")
    test_before = training.print_testing_psnr(test_size=TEST_SIZE)
    
    if PREVIEW and not PERF_NAS:
        training.save_preview_image(training.model_saved,name="../images/last_app_1b.png",mult=1)
        training.save_preview_image(training.model_saved,name="../images/last_app_3b.png",mult=3)
    print()

    train_before = perform_train([LR], iters=MAX_ITERS, size=train_set_size, verbal=VERBAL, mul_approx_func_arr=mul_approx_func_arr) 
    print()

    print("Bingate weights")
    print(training.gate.weight)
    print()
   

    print("Training set (After):")
    train_after = training.forward(train_set_size, optimizer, scheduler, training.model_saved, train=False, size=1, verbal=True)
    
    print()

    area_arr_t = torch.Tensor([1.01, 0.74, 0.03, 0.07, 0.13, 0.07, 0.21, 0.14, 0.5, 0.25, 0.39])
    global TARGET_ACC_VAL
    global TARGET_ACC
    if TARGET_ACC:
        checked = area_arr_t*(((-1)*train_after) > TARGET_ACC_VAL)
        checked[checked==0] = 100
        selected = torch.argmin(checked)
        print(checked)
    else:
        selected = torch.argmin(train_after)

    print("Testing set (After):")
    test_after = training.print_testing_psnr(test_size=TEST_SIZE)
    if PREVIEW and not PERF_NAS:
        training.save_preview_image(training.model_saved,name="../images/last_app_1a.png",mult=1)
        training.save_preview_image(training.model_saved,name="../images/last_app_3a.png",mult=3)
    print()
    if(COPY_FRIENDLY):
        print("Train (Before),Test (Before),Train (After),Test (After)")
        for i in range(len(test_before)):
            print("%.3f" % train_before[i].item()+ ","+ "%.3f" % test_before[i]+ ","+ "%.3f" % train_after[i].item()+ ","+ "%.3f" % test_after[i])

    return train_before, test_before, train_after, test_after, selected

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

if __name__ == '__main__':
    main()