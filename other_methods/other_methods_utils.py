import torch
import numpy as np


def get_times(time, h):
    return torch.tensor(np.arange(0, time + h, h)).to(torch.float32)


def get_RK_change(f, h, *args):
    k1 = f(*args)
    args1 = tuple(a if isinstance(a, tuple) else a + h * k1 / 2 for a in args)
    k2 = f(*args1)
    args2 = tuple(a if isinstance(a, tuple) else a + h * k2 / 2 for a in args)
    k3 = f(*args2)
    args3 = tuple(a if isinstance(a, tuple) else a + h * k3 for a in args)
    k4 = f(*args3)
    return h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
