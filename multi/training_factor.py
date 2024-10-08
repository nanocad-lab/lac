import torch
import torch.nn as nn
import numpy as np
import copy
import torchvision

import torch.nn.functional as F
import utils
import time

trainX_gray, testX_gray = utils.get_cifar_data()
#trainX_gray = torch.reshape(trainX_gray,(1,50000,32,32))
#testX_gray = torch.reshape(testX_gray,(1,50000,32,32))
#print(trainX_gray.size())

def approx_division(value,divisor):
    if divisor == 0:
        out = value
    else:
        out = value/(divisor)
        approx_bitshift = torch.round(torch.log2(divisor))
        out.data = value/(2**approx_bitshift)
    return out

#import applications
# import model_gauss_factor
import ssim_torch_multi

def hinge_loss(input, target):
    hinge_factor = 0.90
    if input<hinge_factor*target:
        return 0
    else:
        return (input-hinge_factor*target)

#print testing psnr for each multiplier separately and weight
def print_testing_psnr(model_saved, file=False, text_file="", verbal=True):
    global testX_gray
    input = testX_gray[:20]
        
    # Calculate target (correct output)
    model_saved.size = 20
    
    target = model_saved.target(input).clone().data
    model_saved.eval()
            
    # Calculate output
    output = model_saved(input)
    
    criterion = model_saved.metric
    # Check NAS setup
    if type(output)==tuple:
        output, cost = output
        # output = output/model_saved.scale
        output_metric_reshaped = torch.reshape(output,target.size())
        loss = -criterion(output_metric_reshaped, target, 255)
        loss_arr = loss
        print("Cost:", cost)
    else:   
        # Calculate loss for each multiplier
        cost = 0
        loss = 0
        loss_arr = [1000]*len(model_saved.mult_list)
        for j in range(len(model_saved.mult_list)): 
            loss1 = -criterion(output[j], target, 255)
            loss_arr[j] = int(float(loss1.data)*1000)/1000
            loss += loss1
    if verbal:
        if file:
            print("Testing loss:", file=text_file)
            print(loss_arr, file=text_file)
            print("Sum of PSNR: {0}".format(int(loss*100)/100), file=text_file)
        else:
            print("Testing loss:")
            print(loss_arr)
            print("Sum of PSNR: {0}".format(int(loss*100)/100))
    return cost, loss_arr
        
def save_preview_image(model,name="../images/preview_image_latest.png",mult=0):
    tensor_preview = utils.get_preview_image()
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

# Define forward pass
def forward(input_size, optimizer, scheduler, model, checkpoint_steps=20, acc=False, train=False, size=1000, verbal=False, timed=False, enable_checkpoints=False, randomize=False, nas=False):
    '''
    optimizer, approximate model, accurate model
    acc=True assumes that func_app is differentiable; acc=False assumes that it's not
    train=False means in training mode, =True means in validation mode
    size=number of data points tested
    timed=True will output time taken for each iteration
    enable_checkpoints=True will save model state and then load the best one of {checkpoint_steps} iterations
    rndomize=True will randomize the dataset if no improvement for {checkpoint steps} iterations
    '''
    global trainX_gray
    global model_saved
    model_saved = model
    
    #Initialize bingate optimizer
    # global gate
    # gate = BinarizeGate(size=len(model.mult_list))
    # optimizer_bingate = torch.optim.SGD(gate.parameters(), lr=2)
    
    # Enable training
    # gate.train()
    
    # Define loss function
    criterion = model.metric
    
    #training set location
    train_set_loc = 0
    
    # Checkpoints
    if nas:
        last_criterions = np.ones(checkpoint_steps+1)*1000
        cost_criterions = np.ones(checkpoint_steps+1)*1000
    else:
        last_criterions = np.ones((len(model.mult_list),checkpoint_steps + 1))*1000
    model_dicts = np.array([model.state_dict()]*(checkpoint_steps + 1))
    #initialize arrays for loss for each multiplier
    loss_arr = [1000]*len(model.mult_list)
    best_arr = [None]*9

    BEST_TRACKER = True
    best_tracker = 0
    best_loss = 100
    
    #Global variables for graphs
    global iii
    iii = 0
    global s_lr
    global s_sum 
    global s_psnr
    # global s_sel
    s_lr = [0]*5000
    model.size = input_size

    stacked = []
    if train:
        model.train()
    else:
        model.eval()
    
    for i in range(size):
        # Generate input
        input = trainX_gray[train_set_loc:train_set_loc+input_size]
        model.nas = nas
        PRINT_BIN_WEIGHTS = False
        if PRINT_BIN_WEIGHTS and nas:
            print(model.bin_gate_0.weight)
            if i>0:
                print(model.bin_gate_0.weight_norm)
                print(model.bin_gate_1.weight_norm)
                print(model.bin_gate_2.weight_norm)
                print(model.bin_gate_3.weight_norm)
                print(model.bin_gate_4.weight_norm)
                print(model.bin_gate_5.weight_norm)
                print(model.bin_gate_6.weight_norm)
                print(model.bin_gate_7.weight_norm)
                print(model.bin_gate_8.weight_norm)
                print()

        
        # Calculate target (correct output)
        target = model.target(input).clone().data
        
        # Calculate output
        end = time.time()
        output = model(input)

        #if train:
        if not model.isNormalized:
            target = approx_division(target,torch.max(target)/255.)+0.00001
            
            if model.nas:
                pass
                #output[0] = approx_division(output[0],torch.max(output[0])/255.)+0.00001
            else:
                for j in range(len(model.mult_list)):
                    with torch.no_grad():
                        model.scale[j] = torch.max(output[j])/255.
                        if torch.isnan(model.scale[j]) or model.scale[j]<=0:
                            model.scale[j] = 1
                        #print(model.scale[j].data)
                    output[j] = approx_division(output[j],model.scale[j].data)+0.00001
                #print("aft: {0}, mean: {1}, max: {2}".format(torch.min(output[j]), torch.mean(output[j]), torch.max(output[j])))
        # print(torch.sum(output))
        # print(output.size())
        # output[0,0,0,0,0,0,0]
        # if train:
        #     if not model.isNormalized:
        #         if nas:
        #             for j in range(len(model.mult_list)):
        #                 with torch.no_grad():
        #                     model.scale[j].data *= torch.max(output[0])/255.#torch.mean(output[j])/torch.mean(target)+0.000001
        #         else:
        #             for j in range(len(model.mult_list)):
        #                 with torch.no_grad():
        #                     model.scale[j].data *= torch.max(output[j])/255.#torch.mean(output[j])/torch.mean(target)+0.000001
        loss = 0
        if nas:
            
            output, cost = output
            print(output.size())
            #output = approx_division(output,torch.max(output)/255.)
            output_metric_reshaped = torch.reshape(output,target.size())
            loss = -criterion(output_metric_reshaped, target, 255)
            print(loss)
            loss_cost = hinge_loss(cost, model.nas_area)
            # print(loss, loss_cost, cost, model.nas_area)
            pure_loss = loss
            loss = loss + loss_cost
            if(model.config[0] != None):
                loss = pure_loss
            
        else:
            loss_arr_tens = [1000]*len(model.mult_list)
            loss_arr = [1000]*len(model.mult_list)
            for j in range(len(model.mult_list)): 
                #output[j] = approx_division(output[j],torch.max(output[j])/255.)
                #output_metric_reshaped = torch.reshape(output[j],target.size())
                loss1 = -criterion(torch.reshape(output[j],target.size()), target, 255)
                #print(loss1)
                loss_arr[j] = float(loss1.data)
                loss_arr_tens[j] = loss1
                loss += loss1

        
        if nas and BEST_TRACKER:
            if pure_loss < best_loss:
                best_loss = pure_loss
                best_tracker = copy.deepcopy(model.state_dict())
                best_arr = [model.bin_gate_0.last_sel,
                model.bin_gate_1.last_sel,
                model.bin_gate_2.last_sel,
                model.bin_gate_3.last_sel,
                model.bin_gate_4.last_sel,
                model.bin_gate_5.last_sel,
                model.bin_gate_6.last_sel,
                model.bin_gate_7.last_sel,
                model.bin_gate_8.last_sel]
                print(best_loss)
        

        if PRINT_BIN_WEIGHTS and not nas:
            print(loss_arr)

        with torch.no_grad():
            model_dicts[i%checkpoint_steps] = copy.deepcopy(model.state_dict())
            if nas:
                last_criterions[i%checkpoint_steps] = loss.item()
                cost_criterions[i%checkpoint_steps] = cost.item()
            else:
                for jj in range(len(model.mult_list)):
                    last_criterions[jj,i%checkpoint_steps] = loss_arr[jj]

        print("ONE")
        if nas:
            stacked = loss
        else:
            stacked = torch.stack(loss_arr_tens,-1)
        
        if train:
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        print("TWO")
        
        # print(model.weight_factor)
        if ((i%checkpoint_steps)==(checkpoint_steps-1)):
            with torch.no_grad():
                if nas:
                    ind_satisfy = cost_criterions<=model.nas_area
                    if (ind_satisfy.max()>0):
                        model_dicts_cost = model_dicts[ind_satisfy]
                        last_criterions_cost = last_criterions[ind_satisfy]
                        model.load_state_dict(model_dicts_cost[np.argmin(last_criterions_cost)])
                        model_dicts[checkpoint_steps] = model.state_dict()
                        model_saved = model.state_dict()
                else:
                    for j in range(len(model.mult_list)):
                        best_ckpt = model_dicts[np.argmin(last_criterions[j])]
                        print(np.min(last_criterions[j]))
                        model.weight[j] = best_ckpt['weight'][j]
                        model.scale[j] = best_ckpt['scale'][j]
                        model.weight_factor[j] = best_ckpt['weight_factor'][j]
                        #loading
                        #model.weight[jj] = model_dicts[np.argmin(last_criterions[jj])][jj]
                        #model.weight_factor[jj] = model_factors[np.argmin(last_criterions[jj])][jj]
                        #model.load_state_dict(model_dicts[np.argmin(last_criterions[j])])
                    model_dicts[checkpoint_steps] = model.state_dict()
                    last_criterions[j,checkpoint_steps] = np.min(last_criterions[j])
                    model_saved = model.state_dict()
            
        #printing
        print("ITER",i)
        if i%10==0:
            if verbal:
                print("Iter: {0}, PSNR of all: {1}\n".format(i,stacked.data.numpy()))
                print("Loss of selected: {0}\n".format(loss))

    # Load best state dict
    print("Loading best state dict")
    model.load_state_dict(model_dicts[checkpoint_steps])
    if best_tracker!=0:
        model.load_state_dict(best_tracker)
    model.bin_gate_0.last_sel = best_arr[0]
    model.bin_gate_1.last_sel = best_arr[1]
    model.bin_gate_2.last_sel = best_arr[2]
    model.bin_gate_3.last_sel = best_arr[3]
    model.bin_gate_4.last_sel = best_arr[4]
    model.bin_gate_5.last_sel = best_arr[5]
    model.bin_gate_6.last_sel = best_arr[6]
    model.bin_gate_7.last_sel = best_arr[7]
    model.bin_gate_8.last_sel = best_arr[8]
    return stacked

loss_pre = 0