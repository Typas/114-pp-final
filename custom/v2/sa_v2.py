import sa_v2_ext

def sa_forward_v2(Q, K, V, scale: float):
    return sa_v2_ext.forward(Q, K, V, float(scale))
