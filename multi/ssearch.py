import torch
import numpy as np
import utils

import training_factor as training
import argparse

import time
torch.manual_seed(1)
import sys
import os
import pickle

LR = 1.

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as inp:
        res = pickle.load(inp)
        return res
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

parser = argparse.ArgumentParser(description='Train parallel nas')

parser.add_argument('-a', '--app', type=str, help='Application')
parser.add_argument('-t', '--target', type=float, default=1.0, help='Target normalized area')
parser.add_argument('-l', '--lr', type=float, default=1.0, help='Learning rate')
parser.add_argument('--end_lr', type=float, default=1e-5, help='End learning rate')

def set_bingate_weights(model, score, config=np.array([0,1,2,3,4,5,6,7,8])):
    ind_satisfy = (model.cost_table < model.nas_area).to(torch.int32)
    if torch.any((model.cost_table < model.nas_area)):
        pre_selected = torch.argmin(ind_satisfy*score)
    else:
        pre_selected = 0 #this isan error - too low area

    INITIAL_VAL_POSITIVE = 100.
    INITIAL_VAL_NEGATIVE = 0.
    print("Before NAS selection: ",pre_selected)
    for i in range(9):
        getattr(model, "bin_gate_{0}".format(i)).fixed = True
        getattr(model, "bin_gate_{0}".format(i)).constant_sel = config[i]


def perform_train(lr_vals, model, iters=40, size = 10, verbal=False, config=np.array([0,1,2,3,4,5,6,7,8])):
    # scores = np.array(lr_vals)
    initial_target = model.nas_area
    scores = []
    for i in range(len(lr_vals)):
        acc = False
        dtype = torch.float32
        input_size = size

        # optimizer = torch.optim.Adam(model.parameters(), lr=lr_vals[i])
        param_list = [{'params': model.weight},
                {'params': model.weight_factor, 'lr': 0, 'factor':0.001}]
        for j in range(9):
            param_list.append({'params': getattr(model,"bin_gate_{0}".format(j)).weight, 'lr':0.3})
        optimizer = torch.optim.Adam(param_list, lr=lr_vals[i])
        # Calculate decay factor
        factor = np.exp(np.log(1e-5/lr_vals[i])/iters)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.95, step_size=1)
        loss_pre = training.forward(input_size, optimizer, scheduler, model, train=False, size=1, nas=True)
        print("Accuracy before training is: {0} , Learning rate: {1}".format(loss_pre, lr_vals[i]))
        iters_pretrain = int(iters*1)
        iters_nas = int(iters*0.4)
        # Pretrain individual multipliers
        # print("Pre training", model.weight.data)
        score = training.forward(input_size, optimizer, scheduler, model, train=False, size=1, acc=acc,
                            verbal=verbal, timed=False, enable_checkpoints = False, nas=False)
        print("Pre-training: ",score)

        print("Post-training: ",score)
        set_bingate_weights(model, score, config=config)
        # Perform NAS calculations
        score = training.forward(input_size, optimizer, scheduler, model, train=True, size=iters, acc=acc,
                            verbal=verbal, timed=False, enable_checkpoints = True, nas=True, checkpoint_steps=20)
        scores.append(score.item())
        print(scores)
        # print("Post training", model.weight.data)
        # training.save_preview_image(model,name="../images/DRUM6_demo_sobel_after.png",mult=0)
    return np.stack(scores,0)

def run1(target, iters=200, images=10, config=np.array([0,1,2,3,4,5,6,7,8])):
    end = time.time()
    global args
    global mul_approx_func_arr
    args = parser.parse_args()
    application = "Gaussian"
    torch.manual_seed(0)
    # mul_approx_func_arr = [utils.decode_multiplier(multiplier)]
    # Gaussian 6,7 - problematic #0, 5,
    # mul_approx_func_arr = [0,1,2,3,4,5,7,10,11,12] [0,3,5,12] ,6,10,11
    
    #[0,1,2,3,4,5,6,7,10,11,12]#[1, 2, 7, 6]#, 3, 4, 11, 10]#[10,11]#[0,3,5,10,11,12]#Multiplier 0 can be removed
   #NAS python .\train_graph.py -a Gaussian -t 100 -l 0.5
    #pretrain python .\train_graph.py -a Gaussian -t 100 -l 1.
    # mul_approx_func_arr = [1,2,4,6,10,11,13]
    # mul_approx_func_arr = [1,2,4,6,10,11,13]
    
    #print("Training set (Before):")
    model = utils.generate_model(application, mul_approx_func_arr, nas=True, nas_area=target)
    model.config = config

    optimizer = torch.optim.Adam(model.parameters(), lr=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)
    loss_pre = training.forward(100, optimizer, scheduler, model, train=False, size=1, verbal=False, nas=False)
    #print("Testing set (Before):")
    training.print_testing_psnr(model, verbal=False)
    
    output_psnr = perform_train([LR], model, iters=iters, size=images, verbal=False, config=config)
    model.bin_gate_0.fixed = True
    model.bin_gate_1.fixed = True
    model.bin_gate_2.fixed = True
    model.bin_gate_3.fixed = True
    model.bin_gate_4.fixed = True
    model.bin_gate_5.fixed = True
    model.bin_gate_6.fixed = True
    model.bin_gate_7.fixed = True
    model.bin_gate_8.fixed = True

    cost, test_psnr = training.print_testing_psnr(model, verbal=False)
    print(training.forward(100, optimizer, scheduler, model, train=False, size=1, verbal=False, nas=True))
    print(type(cost))
    #print()
    #print("Training set (After):")
    #print(output_psnr)
    #print("Testing set (After):")
    #print(test_psnr)
    for i in range(9):
        ind=torch.argmax(getattr(model, "bin_gate_{0}".format(i)).weight)
        print(model.mult_list[ind], end=" ")
    print()
    #print("Total time")
    #print(time.time()-end)
    return cost, output_psnr, test_psnr

def run2(multn=11, picktype="order", default_fill="-1"):#lr = 1
    global mul_approx_func_arr
    mul_approx_func_arr = [1, 2, 0, 5, 12, 7, 6, 3, 4, 11, 10]
    BLOCK_PRINT = True
    final_res = 0
    config_g = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1])
    pre_mult_set = np.arange(0,multn)#equal to range(arg2) #np.array([0,1,2,3,4,5,6,7,8,9,10])
    global AVERAGE_AREA_LIMIT
    area_cutoff = AVERAGE_AREA_LIMIT
    mult_set = np.array([])
    for mult in pre_mult_set:
        cur = utils.multiplier_area(mul_approx_func_arr[mult])
        if cur < area_cutoff:
            mult_set = np.append(mult_set, [mult])
    print("Effective mul",mult_set)

    stage_res = np.zeros(mult_set.size)
    for stage in range(0,9):
        if picktype == "random":
            next = np.random.randint(0, 9)
            while config_g[next]!=-1:
                next = np.random.randint(0, 9)
            print("rand",next)
        if picktype == "order":
            next = stage

        cost_table = [-100.0]*mult_set.size

        for i,mult in enumerate(mult_set):#[0.08, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.2]:
            config = config_g.copy()
            config[next] = mult
            if default_fill == "cur":
                for j,m in enumerate(config):
                    if m == -1: config[j] = mult
            print("Config: ",config)
            if BLOCK_PRINT:
                blockPrint()
            cost, train_metric, metric = run1(1000, iters=21, images=1, config=config)
            cost_table[int(i)] = cost
            if BLOCK_PRINT:
                enablePrint()
            stage_res[i] = metric.item()
        print(stage_res)
        
        #print(np.argmin(stage_res))
        final_res = np.min(stage_res)
        config_g[next] = mult_set[np.argmin(stage_res)]
        #print(cost.item(), " ", train_metric.item(), " ", metric.item())
    return cost_table[np.argmin(stage_res)], config_g, final_res #should be .item

def single():
    start = time.time()
    BLOCK_PRINT = False
    if BLOCK_PRINT:
        blockPrint()
    cost, config, metric = run2(multn=11, picktype="random", default_fill="cur")
    area_av = 0
    area_arr = [1.01, 0.74, 0.03, 0.07, 0.13, 0.07, 0.21, 0.14, 0.5, 0.25, 0.39]
    for mult in config:
        area_av += area_arr[mult]/9
    
    if BLOCK_PRINT:
        enablePrint()
    print("cost", "%.3f" % area_av, "metric ", "%.3f" % metric, "config", config, "time ","%.3f" % (time.time()-start))

def main():
    global AVERAGE_AREA_LIMIT
    mode = 1
    if mode == 0:
        AVERAGE_AREA_LIMIT = 0.5
        single()
    if mode == 1:
        for thres in np.arange(0.1, 1.2, 0.1):
            AVERAGE_AREA_LIMIT = thres
            single()
    if mode == 2:
        cost, train_metric, metric = run1(1000, iters=21, images=1, config=np.array([0,1,2,3,4,5,6,7,8]))
if __name__ == '__main__':
    main()