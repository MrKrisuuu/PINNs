from constants.initial_conditions import get_initial_conditions
import torch


def get_LV_start_c():
    (X, Y, params) = get_initial_conditions("LV")
    (a, b, c, d) = params
    return - a*torch.log(torch.tensor(Y[0])) + b*Y[0] + c*X[0] - d*torch.log(torch.tensor(X[0]))


def get_LV_c(X, Y):
    (_, _, params) = get_initial_conditions("LV")
    (a, b, c, d) = params
    X = torch.where(X < 0, torch.tensor(0.001), X)
    Y = torch.where(Y < 0, torch.tensor(0.001), Y)
    return - a*torch.log(Y) + b*Y + c*X - d*torch.log(X)