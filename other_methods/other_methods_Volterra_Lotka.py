import torch
import scipy

from other_methods.other_methods_utils import get_RK_change, get_times


def get_starting_values():
    X = [1]
    Y = [1]
    params = (1, 1, 1, 2)
    return (X, Y, params)


def fX(x, y, params):
    a, b, c, d = params
    return (a - b * y) * x


def fY(x, y, params):
    a, b, c, d = params
    return (c * x - d) * y


def VL_equation(values, previous, h, params):
    x_n1, y_n1 = values
    x_n, y_n = previous
    a, b, c, d = params
    X = x_n1 - x_n - h * (a - b * y_n1) * x_n1
    Y = y_n1 - y_n - h * (c * x_n1 - d) * y_n1
    return [X, Y]


def euler_VL(time, h=0.01):
    X, Y, params = get_starting_values()
    dX = fX(X[-1], Y[-1], params)
    dY = fY(X[-1], Y[-1], params)
    times = get_times(time, h)
    for t in times[1:]:
        X.append(X[-1] + dX * h)
        Y.append(Y[-1] + dY * h)
        dX = fX(X[-1], Y[-1], params)
        dY = fY(X[-1], Y[-1], params)
    return torch.tensor(X), torch.tensor(Y), times


def semi_VL(time, h=0.01):
    # Is it possible to implement it?
    pass


def implicit_VL(time, h=0.01):
    X, Y, params = get_starting_values()
    times = get_times(time, h)
    for t in times[1:]:
        prev = (X[-1], Y[-1])
        x, y = scipy.optimize.fsolve(VL_equation, prev, args=(prev, h, params))
        X.append(x)
        Y.append(y)
    return torch.tensor(X), torch.tensor(Y), times


def RK_VL(time, h=0.01):
    X, Y, params = get_starting_values()
    times = get_times(time, h)
    for t in times[1:]:
        dX = get_RK_change(fX, h, X[-1], Y[-1], params)
        dY = get_RK_change(fY, h, X[-1], Y[-1], params)

        X.append(X[-1] + dX)
        Y.append(Y[-1] + dY)
    return torch.tensor(X), torch.tensor(Y), times