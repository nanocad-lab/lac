import torch
import pyximport; pyximport.install()
import mult_approx

class QuantizeGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
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

    # Gradients (accurate multiplication)
    #out = ((in1.type(in_dtype))*(in2.type(in_dtype))).type(out_dtype)
    out = QuantizeGrad.apply((in1*in2).squeeze(0))
        
    # Special sign handling for some mulipliers
    if mult_approx_index == -1:
        return (in1.type(torch.int32)*in2.type(torch.int32)).type(torch.int32)
    if True: #mult_approx_index in [3,4,8,9,10,11]:
        out_data = torch.sign(in1)*torch.sign(in2)*mult_approx.mult_approx(abs(in1.type(in_dtype)), abs(in2.type(in_dtype)), mult_approx_index).type(out_dtype)
    else:
        out_data = mult_approx.mult_approx(in1.type(in_dtype), in2.type(in_dtype), mult_approx_index).type(out_dtype)

    # Use approxiamte data, accurate gradients
    out.data = out_data.to(torch.float32)
    return out