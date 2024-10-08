import torch
import numpy as np
import utils

import training_factor as training
import argparse

import time
torch.manual_seed(0)
import sys
import os
import pickle

LR = 1

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

def set_bingate_weights(model, score):
    ind_satisfy = (model.cost_table < model.nas_area).to(torch.int32)
    if torch.any((model.cost_table < model.nas_area)):
        pre_selected = torch.argmin(ind_satisfy*score)
    else:
        pre_selected = 0 #this isan error - too low area

    INITIAL_VAL_POSITIVE = 7.
    INITIAL_VAL_NEGATIVE = 0.
    print("Before NAS selection: ",pre_selected)
    model.bin_gate_0.weight.data *= INITIAL_VAL_NEGATIVE
    model.bin_gate_0.weight.data[pre_selected] = INITIAL_VAL_POSITIVE
    model.bin_gate_1.weight.data *= INITIAL_VAL_NEGATIVE
    model.bin_gate_1.weight.data[pre_selected] = INITIAL_VAL_POSITIVE
    model.bin_gate_2.weight.data *= INITIAL_VAL_NEGATIVE
    model.bin_gate_2.weight.data[pre_selected] = INITIAL_VAL_POSITIVE
    model.bin_gate_3.weight.data *= INITIAL_VAL_NEGATIVE
    model.bin_gate_3.weight.data[pre_selected] = INITIAL_VAL_POSITIVE
    model.bin_gate_4.weight.data *= INITIAL_VAL_NEGATIVE
    model.bin_gate_4.weight.data[pre_selected] = INITIAL_VAL_POSITIVE
    model.bin_gate_5.weight.data *= INITIAL_VAL_NEGATIVE
    model.bin_gate_5.weight.data[pre_selected] = INITIAL_VAL_POSITIVE
    model.bin_gate_6.weight.data *= INITIAL_VAL_NEGATIVE
    model.bin_gate_6.weight.data[pre_selected] = INITIAL_VAL_POSITIVE
    model.bin_gate_7.weight.data *= INITIAL_VAL_NEGATIVE
    model.bin_gate_7.weight.data[pre_selected] = INITIAL_VAL_POSITIVE
    model.bin_gate_8.weight.data *= INITIAL_VAL_NEGATIVE
    model.bin_gate_8.weight.data[pre_selected] = INITIAL_VAL_POSITIVE

def perform_train(lr_vals, model, iters=40, size = 10, verbal=False):
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

        # IS_SAVED = True
        # if not IS_SAVED:
        score = training.forward(input_size, optimizer, scheduler, model, train=True, size=iters_pretrain, acc=acc,
                                verbal=verbal, timed=False, enable_checkpoints = False, nas=False)
        #     torch.save({
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'score': score
        #     }, 'saved_torch')
        # else:
        #     checkpoint = torch.load('saved_torch')
        #     model.load_state_dict(checkpoint['model_state_dict'])
        #     model.nas_area = initial_target
        #     #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #     score = checkpoint['score']
        #     #for g in optimizer.param_groups:
        #         #print(g['params'][0].size())
        #         #g['lr'] = 0.001
        print("Post-training: ",score)
        set_bingate_weights(model, score)
        # Perform NAS calculations
        score = training.forward(input_size, optimizer, scheduler, model, train=True, size=iters_nas, acc=acc,
                            verbal=verbal, timed=False, enable_checkpoints = False, nas=True, checkpoint_steps=70)
        scores.append(score.item())
        print(scores)
        # print("Post training", model.weight.data)
        # training.save_preview_image(model,name="../images/DRUM6_demo_sobel_after.png",mult=0)
    return np.stack(scores,0)

def main():
    BLOCK_PRINT = True
    for target in [0.3, 1.0]:#[0.08, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.2]:
        if BLOCK_PRINT:
            blockPrint()
        cost, train_metric, metric = run1(target, iters=200, images=100)
        if BLOCK_PRINT:
            enablePrint()
        print(cost.item(), " ", train_metric.item(), " ", metric.item())

def run1(target, iters=200, images=10):
    end = time.time()
    global args
    global mul_approx_func_arr
    args = parser.parse_args()
    application = "Gaussian"
    torch.manual_seed(0)
    mul_approx_func_arr = [1, 2, 0, 5, 12, 7, 6, 3, 4, 11, 10]
    #NAS python .\train_graph.py -a Gaussian -t 100 -l 0.5
    #pretrain python .\train_graph.py -a Gaussian -t 100 -l 1.

    #print("Training set (Before):")
    model = utils.generate_model(application, mul_approx_func_arr, nas=True, nas_area=target)
    optimizer = torch.optim.Adam(model.parameters(), lr=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)
    loss_pre = training.forward(100, optimizer, scheduler, model, train=False, size=1, verbal=False, nas=False)
    #print("Testing set (Before):")
    training.print_testing_psnr(model, verbal=False)
    
    output_psnr = perform_train([LR], model, iters=iters, size=images, verbal=False)
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

if __name__ == '__main__':
    main()