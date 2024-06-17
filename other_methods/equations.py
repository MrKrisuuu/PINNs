import torch


def SIR(t, y, params):
    b, g, N = params
    return torch.tensor([-b / N * y[0] * y[1],
                         b / N * y[0] * y[1] - g * y[1],
                         g * y[1]])


def Kepler(t, y, params):
    _ = params
    r = (y[0]**2 + y[1]**2)**(1/2)
    return torch.tensor([y[2],
                         y[3],
                         -y[0] / r,
                         -y[1] / r])


def LV(t, y, params):
    a, b, c, d = params
    return torch.tensor([(a - b * y[1]) * y[0],
                         (c * y[0] - d) * y[1]])