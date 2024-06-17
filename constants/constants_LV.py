from constants.initial_conditions import get_initial_conditions
import torch


def get_LV_start_c():
    (y, params) = get_initial_conditions("LV")
    (a, b, c, d) = params
    return - a*torch.log(y[1]) + b*y[1] + c*y[0] - d*torch.log(y[0])


def get_LV_c(y):
    (_, params) = get_initial_conditions("LV")
    (a, b, c, d) = params
    y = torch.where(y < 0, torch.tensor(0.001), y)
    return - a*torch.log(y[:, 1]) + b*y[:, 1] + c*y[:, 0] - d*torch.log(y[:, 0])