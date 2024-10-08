#!/usr/bin/env python
# coding: utf-8


import torch

def forward2k(input, size):
    l1 = 0.5
    l2 = 0.5
    result = torch.zeros((1,size,2))
    for i in range(size):
        result[0,i,0] = l1 * torch.cos(input[0,i,0]) + l2 * torch.cos(input[0,i,0] + input[0,i,1])
        result[0,i,1] = l1 * torch.sin(input[0,i,0]) + l2 * torch.sin(input[0,i,0] + input[0,i,1])
    return result

file = "../axbench_data/in2k/data_in2k_test.data"
data = [x.split(' ') for x in open(file).readlines()]
colomns = [x[0].replace('\n','').split('\t') for x in data[1:]]
testX_gray = torch.zeros((1,len(colomns),2))
for i in range(len(colomns)):
    testX_gray[0,i,0] = float(colomns[i][0])
    testX_gray[0,i,1] = float(colomns[i][1])

file = "../axbench_data/in2k/data_in2k_train.data"
data = [x.split(' ') for x in open(file).readlines()]
colomns = [x[0].replace('\n','').split('\t') for x in data[1:]]
trainX_gray = torch.zeros((1,len(colomns),2))
for i in range(len(colomns)):
    trainX_gray[0,i,0] = float(colomns[i][0])
    trainX_gray[0,i,1] = float(colomns[i][1])

testX_gray = forward2k(testX_gray, 100000)
trainX_gray = forward2k(trainX_gray, 100000)


torch.save(trainX_gray, '../axbench_data/in2k/train_tensor.pt')
torch.save(testX_gray, '../axbench_data/in2k/test_tensor.pt')

print(torch.load('../axbench_data/in2k/train_tensor.pt').size())
print(torch.load('../axbench_data/in2k/test_tensor.pt').size())



