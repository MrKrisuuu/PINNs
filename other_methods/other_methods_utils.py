import torch
import numpy as np


def get_times(time, h):
    return torch.tensor(np.arange(0, time + h, h)).to(torch.float32)


def get_RK_change(f, h, *args):
    k1 = h * f(*args)
    args1 = tuple(a if isinstance(a, tuple) else a + k1 / 2 for a in args)
    k2 = h * f(*args1)
    args2 = tuple(a if isinstance(a, tuple) else a + k2 / 2 for a in args)
    k3 = h * f(*args2)
    args3 = tuple(a if isinstance(a, tuple) else a + k3 for a in args)
    k4 = h * f(*args3)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6
