import sa_v4_ext

def sa_forward_v4(Q, K, V, scale: float):
    return sa_v4_ext.forward(Q, K, V, float(scale))
