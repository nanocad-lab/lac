import torch
import numpy as np
import utils

import training_factor as training
import argparse

import time
torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Train parallel nas')

parser.add_argument('-a', '--app', type=str, help='Application')
parser.add_argument('-t', '--target', type=float, default=1.0, help='Target normalized area')
parser.add_argument('-l', '--lr', type=float, default=1.0, help='Learning rate')
parser.add_argument('--end_lr', type=float, default=1e-5, help='End learning rate')

def perform_train(lr_vals, model, iters=40, size = 10, verbal=False):
    # scores = np.array(lr_vals)
    scores = []
    for i in range(len(lr_vals)):
        acc = False
        dtype = torch.float32
        input_size = size
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr_vals[i])
        param_list = [{'params': model.weight},
                {'params': model.weight_factor, 'lr': 100}]
        for j in range(9):
            param_list.append({'params': getattr(model,"bin_gate_{0}".format(j)).weight, 'lr':100})
        optimizer = torch.optim.Adam(param_list, lr=lr_vals[i])
        # Calculate decay factor
        factor = np.exp(np.log(1e-5/lr_vals[i])/iters)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=factor, verbose=False)
        loss_pre = training.forward(input_size, optimizer, scheduler, model, train=False, size=1, nas=True)
        print("Accuracy before training is: {0} , Learning rate: {1}".format(loss_pre, lr_vals[i]))
        iters_pretrain = int(iters*0.8)
        iters_nas = int(iters*0.2)
        # Pretrain individual multipliers
        # print("Pre training", model.weight.data)
        score = training.forward(input_size, optimizer, scheduler, model, train=True, size=iters_pretrain, acc=acc,
                            verbal=verbal, timed=False, enable_checkpoints = False, nas=False)
        # Perform NAS calculations
        score = training.forward(input_size, optimizer, scheduler, model, train=True, size=iters_nas, acc=acc,
                            verbal=verbal, timed=False, enable_checkpoints = False, nas=True)
        scores.append(score.item())
        # print("Post training", model.weight.data)
        # training.save_preview_image(model,name="../images/DRUM6_demo_sobel_after.png",mult=0)
    return np.stack(scores,0)

def main():
    end = time.time()
    global args
    args = parser.parse_args()
    target = args.target
    application = args.app
    torch.manual_seed(0)

    mul_approx_func_arr = [0,1,2,3,4,5,7,10,11,12]


    print("Training set (Before):")
    model = utils.generate_model(application, mul_approx_func_arr, nas=True, nas_area=target)
    optimizer = torch.optim.Adam(model.parameters(), lr=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)
    loss_pre = training.forward(100, optimizer, scheduler, model, train=False, size=1, verbal=True, nas=True)
    print("Testing set (Before):")
    training.print_testing_psnr(model, verbal=True)
    
    output_psnr = perform_train([args.lr], model, iters=200, size=100, verbal=False)
    test_psnr = training.print_testing_psnr(model, verbal=False)

    print()
    print("Training set (After):")
    print(output_psnr)
    print("Testing set (After):")
    print(test_psnr)
    for i in range(9):
        ind=torch.argmax(getattr(model, "bin_gate_{0}".format(i)).weight)
        print(model.mult_list[ind])
    print("Total time")
    print(time.time()-end)

if __name__ == '__main__':
    main()