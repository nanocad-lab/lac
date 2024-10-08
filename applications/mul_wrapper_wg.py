import torch
import pyximport; pyximport.install()
import mult_approx

#def wg(x):
#    return torch.round(x-0.5) + (x-0.5-torch.round(x-0.5))**7*64

# def wg(x):
#     return torch.round(x-0.5) + (x-0.5-torch.round(x-0.5))**5*16

#def wg(x):
#   return torch.round(x-0.5) + (x-0.5-torch.round(x-0.5))**3*4

def wg(x):
    return x

def round_grad(x):
     full_res = wg(x)
     round_res = torch.round(x)
     full_res.data = round_res
     return full_res

def mul_wrapper(in1, in2, mult_approx_index):
    
    # Assigning proper datatype to input/output
    in_dtype = torch.int16
    out_dtype = torch.int32
    if mult_approx_index in []:
        out_dtype = torch.int8
    if mult_approx_index in [7]:
        in_dtype = torch.int8
        out_dtype = torch.int8
    if mult_approx_index in [6,12]:
        in_dtype = torch.int8
        out_dtype = torch.int16
    if mult_approx_index in []:
        out_dtype = torch.int16
        
    full_res = torch.reshape(torch.mul(wg(in1),wg(in2)).type(torch.float),(1,-1))
    # Special sign handling for some mulipliers
    if mult_approx_index == -1:
        return (in1.type(torch.int32)*in2.type(torch.int32)).type(torch.int32)
    if mult_approx_index in [3,4,8,9,10,11]: #10/10 fix - add 0,5
        approx_res = torch.sign(in1)*torch.sign(in2)*mult_approx.mult_approx(abs(in1.type(in_dtype)), abs(in2.type(in_dtype)), mult_approx_index).type(out_dtype)
    else:
        approx_res = mult_approx.mult_approx(in1.type(in_dtype), in2.type(in_dtype), mult_approx_index).type(out_dtype)
    
    full_res.data = torch.reshape(approx_res.type(torch.float),(1,-1))
    return full_res