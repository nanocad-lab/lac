import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image
import models

def get_cifar_data():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())     
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())    
    rgb2g = torch.tensor([0.2989, 0.587, 0.114])

    trainX_gray = torch.zeros(len(trainset), 1, 32, 32)
    testX_gray = torch.zeros(len(testset), 1, 32, 32)
    # trainX_gray = torch.zeros((1,1000,32,32))
    # testX_gray = torch.zeros((1,1000,32,32))

    for i in range(1000):
        trainX_gray[i] = torch.round((trainset[i][0]*rgb2g.view(-1,1,1)).sum(0)*255)
    for i in range(1000):
        testX_gray[i] = torch.round((testset[i][0]*rgb2g.view(-1,1,1)).sum(0)*255)

    # for i in range(1000):
    #     trainX_gray[0,i] = torch.round((trainset[i][0][0]*rgb2g[0]+trainset[i][0][1]*rgb2g[1]+trainset[i][0][2]*rgb2g[2])*255)
    # for i in range(1000):
    #     testX_gray[0,i] = torch.round((testset[i][0][0]*rgb2g[0]+testset[i][0][1]*rgb2g[1]+testset[i][0][2]*rgb2g[2])*255)
    return trainX_gray, testX_gray

def get_preview_image():
    image_preview = Image.open('../images/cameraman.tif')

    # Define a transform to convert the image to tensor
    transform = transforms.ToTensor()

    # Convert the image to PyTorch tensor
    tensor_preview = transform(image_preview)*255.
    return tensor_preview

def decode_multiplier(multiplier):
    if multiplier=='mul8u_JV3':
        mul_code = 0
    elif multiplier=='mul16s_GK2':
        mul_code = 1
    elif multiplier=='mul16s_GAT':
        mul_code = 2
    elif multiplier=='EMT':
        mul_code = 3
    elif multiplier=='EMT_16':
        mul_code = 4
    elif multiplier=='mul8u_FTA':
        mul_code = 5
    elif multiplier=='mul8s_1KVL':
        mul_code = 6
    elif multiplier=='mul8s_1KR3':
        mul_code = 7
    elif multiplier=='kulkarni_16bit':
        mul_code = 8
    elif multiplier=='kulkarni_8bit':
        mul_code = 9
    elif multiplier=='DRUM_16bit':
        mul_code = 10
    elif multiplier=='DRUM_16bit_4':
        mul_code = 11
    elif multiplier=='mul8u_185Q':
        mul_code = 12
    elif multiplier=='EMT_s':
        mul_code = 13
    return mul_code

def multiplier_area(mul_code: int):
    if mul_code==-1:
        return 0
    code_map = [0.03, 1.01, 0.74, 0.14, 0.50, 0.07, 0.21, 0.07, 0, 0, 0.39, 0.25, 0.13,0.14]
    return code_map[mul_code]

def generate_model(application,mul_approx_func_arr, nas=False, nas_area=1.0):
    if application=='Sobel':
        forward_model = models.Sobel_Forward_Model(100,mul_approx_func_arr=mul_approx_func_arr, nas=nas, nas_area=nas_area)
    elif application=='Gaussian':
        forward_model = models.Gaussian_Forward_Model(100, mul_approx_func_arr=mul_approx_func_arr, nas=nas, nas_area=nas_area)
    elif application=='Sharp':
        forward_model = models.Laplacian_Forward_Model(100, mul_approx_func_arr=mul_approx_func_arr, nas=nas, nas_area=nas_area)
    elif application=='DCT':
        forward_model = models.DCT_Forward_Model(100, mul_approx_func_arr=mul_approx_func_arr, nas=nas)
    return forward_model