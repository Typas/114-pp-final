import sa_v3_ext

def sa_forward_v3(Q, K, V, scale: float):
    return sa_v3_ext.forward(Q, K, V, float(scale))
