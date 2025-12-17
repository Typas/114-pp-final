import torch
import sa_v1_ext

def sa_forward_v1(Q, K, V, scale: float):
    # expects fp16 CUDA contiguous
    return sa_v1_ext.forward(Q, K, V, float(scale))
