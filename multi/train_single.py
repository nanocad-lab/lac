import torch
import numpy as np
import utils
import time

import training_factor as training
import argparse

parser = argparse.ArgumentParser(description='Train a single multiplier')

parser.add_argument('-a', '--app', type=str, help='Application')
parser.add_argument('-m', '--mult', type=str, help='Multiplier')
parser.add_argument('-l', '--lr', type=float, default=1.0, help='Learning rate')
parser.add_argument('--end_lr', type=float, default=1e-5, help='End learning rate')

effective_multipliers = [0,1,3,5,6,8,9,10,11,12]

def perform_train(lr_vals, model, iters=40, size = 10, verbal=False):
    # scores = np.array(lr_vals)
    scores = []
    for i in range(len(lr_vals)):
        acc = False
        dtype = torch.float32
        input_size = size
        # model = model_sobel_factor.Forward_Model(size,mul_approx_func_arr=mul_approx_func_arr)
        # training.save_preview_image(model,name="../images/DRUM6_demo_sobel_before.png",mult=0)
        # model = model_sobel_factor.Forward_Model(size,mul_approx_func_arr=mul_approx_func_arr)
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr_vals[i], momentum=0.9)
        optimizer = torch.optim.Adam([
                {'params': model.weight},
                {'params': model.weight_factor, 'lr': 100}
            ], lr=lr_vals[i])
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)
        # Calculate decay factor
        factor = np.exp(np.log(1e-5/lr_vals[i])/iters)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=factor, verbose=False)
        loss_pre = training.forward(input_size, optimizer, scheduler, model, train=False, size=1)
        print("Accuracy before training is: {0} , Learning rate: {1}".format(loss_pre, lr_vals[i]))
        # print("Pre training", model.weight.data)
        score = training.forward(input_size, optimizer, scheduler, model, train=True, size=iters, acc=acc,
                            verbal=verbal, timed=False, enable_checkpoints = False) 
        # print("Post training", model.weight.data)
        scores.append(score)
        # training.save_preview_image(model,name="../images/DRUM6_demo_sobel_after.png",mult=0)
    return torch.stack(scores,0).detach()

def main():
    global args
    end = time.time()
    args = parser.parse_args()
    multiplier = args.mult
    application = args.app
    mul_approx_func_arr = [utils.decode_multiplier(multiplier)]
    # mul_approx_func_arr = effective_multipliers
    print("Training set (Before):")
    model = utils.generate_model(application, mul_approx_func_arr)
    optimizer = torch.optim.Adam([
                    {'params': model.weight},
                    {'params': model.weight_factor, 'lr': 1, 'factor':0.1}
                ], lr=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)
    loss_pre = training.forward(100, optimizer, scheduler, model, train=False, size=1, verbal=True)
    print("Testing set (Before):")
    training.print_testing_psnr(model, verbal=True)
    output_psnr = [0]*len(mul_approx_func_arr)
    test_psnr = [0]*len(mul_approx_func_arr)

    # for ii,mult in enumerate(mul_approx_func_arr):
    #     mul_approx_func_arr = [mult]
    output_psnr = perform_train([args.lr], model, iters=200, size=100, verbal=False) 
    test_psnr = training.print_testing_psnr(model, verbal=False)

    print()
    print("Training set (After):")
    print(output_psnr)
    print("Testing set (After):")
    print(test_psnr)
    print("Total time:", time.time()-end)

if __name__ == '__main__':
    main()