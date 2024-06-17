from constants.initial_conditions import get_initial_conditions
import torch


def get_SIR_start_sum():
    (y, params) = get_initial_conditions("SIR")
    (_, _, N) = params
    return N


def get_SIR_sum(y):
    return y[:, 0] + y[:, 1] + y[:, 2]


def get_SIR_start_constant():
    (y, params) = get_initial_conditions("SIR")
    (b, g, N) = params
    return g*torch.log(y[0]) + b*y[2]


def get_SIR_constant(y):
    (_, params) = get_initial_conditions("SIR")
    (b, g, N) = params
    return g*torch.log(y[:, 0]) + b*y[:, 2]