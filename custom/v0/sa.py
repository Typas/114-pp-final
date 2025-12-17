import torch
import sa_cuda_ext

def sa_forward(Q, K, V, scale: float):
    # enforce contiguous + fp16 for v0
    if Q.dtype != torch.float16: Q = Q.half()
    if K.dtype != torch.float16: K = K.half()
    if V.dtype != torch.float16: V = V.half()
    Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous()
    return sa_cuda_ext.forward(Q, K, V, float(scale))
