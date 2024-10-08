import torch
import pyximport; pyximport.install()
import mult_approx

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
    if mult_approx_index in [-1]:
        in_dtype = torch.int32
        out_dtype = torch.int64
        

    # Special sign handling for some mulipliers
    if mult_approx_index == -1:
        return in1.type(torch.int32)*in2.type(torch.int32).type(torch.int64)
    if mult_approx_index in [3,4,8,9,10,11,12]: #10/10 fix - add 0,5
        return torch.sign(in1)*torch.sign(in2)*mult_approx.mult_approx(abs(in1.type(in_dtype)), abs(in2.type(in_dtype)), mult_approx_index).type(out_dtype)
    else:
        return mult_approx.mult_approx(in1.type(in_dtype), in2.type(in_dtype), mult_approx_index).type(out_dtype)