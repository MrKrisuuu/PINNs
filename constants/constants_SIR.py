from constants.initial_conditions import get_initial_conditions
import torch


def get_SIR_start_sum():
    (_, _, _, params) = get_initial_conditions("SIR")
    (_, _, N) = params
    return N


def get_SIR_sum(S, I, R):
    return S + I + R


def get_SIR_start_constant():
    (S, I, R, params) = get_initial_conditions("SIR")
    (b, y, N) = params
    return y*torch.log(torch.tensor(S[0])) + b*R[0]


def get_SIR_constant(S, I, R):
    (_, _, _, params) = get_initial_conditions("SIR")
    (b, y, N) = params
    return y*torch.log(S) + b*R