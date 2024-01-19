import torch
import scipy

from other_methods.other_methods_utils import get_RK_change, get_times
from constants.initial_conditions import get_initial_conditions


def fdX(x, y):
    r = (x**2 + y**2)**(1/2)
    return - x / r**3


def fdY(x, y):
    r = (x**2 + y**2)**(1/2)
    return - y / r**3


def fX(x):
    return x


def fY(y):
    return y


def Kepler_equation(values, previous, h):
    x_n1, y_n1, dx_n1, dy_n1 = values
    x_n, y_n, dx_n, dy_n = previous
    X = x_n1 - x_n - h * dx_n1
    Y = y_n1 - y_n - h * dy_n1
    r = (x_n1**2 + y_n1**2)**(1/2)
    dX = dx_n1 - dx_n + h * x_n1 / r**3
    dY = dy_n1 - dy_n + h * y_n1 / r**3
    return [X, Y, dX, dY]


def euler_Kepler(time, h=0.01):
    X, Y, dX, dY = get_initial_conditions("Kepler")
    times = get_times(time, h)
    for t in times[1:]:
        ddX = fdX(X[-1], Y[-1])
        ddY = fdY(X[-1], Y[-1])
        X.append(X[-1] + fX(dX) * h)
        Y.append(Y[-1] + fY(dY) * h)
        dX = dX + ddX * h
        dY = dY + ddY * h
    return torch.tensor(X), torch.tensor(Y), times


def semi_Kepler(time, h=0.01):
    X, Y, dX, dY = get_initial_conditions("Kepler")
    times = get_times(time, h)
    for t in times[1:]:
        ddX = fdX(X[-1], Y[-1])
        ddY = fdY(X[-1], Y[-1])
        dX = dX + ddX * h
        dY = dY + ddY * h
        X.append(X[-1] + fY(dX) * h)
        Y.append(Y[-1] + fY(dY) * h)
    return torch.tensor(X), torch.tensor(Y), times


def implicit_Kepler(time, h=0.01):
    X, Y, dX, dY = get_initial_conditions("Kepler")
    times = get_times(time, h)
    for t in times[1:]:
        prev = (X[-1], Y[-1], dX, dY)
        (x, y, dx, dy), xd, _, _ = scipy.optimize.fsolve(Kepler_equation, prev, args=(prev, h), full_output=True)
        X.append(x)
        Y.append(y)
        dX = dx
        dY = dy
    return torch.tensor(X), torch.tensor(Y), times


def RK_Kepler(time, h=0.01):
    X, Y, dX, dY = get_initial_conditions("Kepler")
    times = get_times(time, h)
    for t in times[1:]:
        d_X = get_RK_change(fX, h, dX)
        d_Y = get_RK_change(fY, h, dY)
        d_dX = get_RK_change(fdX, h, X[-1], Y[-1])
        d_dY = get_RK_change(fdY, h, X[-1], Y[-1])

        X.append(X[-1] + d_X)
        Y.append(Y[-1] + d_Y)
        dX = dX + d_dX
        dY = dY + d_dY
    return torch.tensor(X), torch.tensor(Y), times


def Verlet_Kepler(time, h=0.01):
    X, Y, dX, dY = get_initial_conditions("Kepler")
    times = get_times(time, h)
    for t in times[1:]:
        nextX = X[-1] + dX * h + fdX(X[-1], Y[-1]) / 2 * h * h
        nextY = Y[-1] + dY * h + fdY(X[-1], Y[-1]) / 2 * h * h

        nextdX = dX + (fdX(nextX, nextY) + fdX(X[-1], Y[-1])) / 2 * h
        nextdY = dY + (fdY(nextX, nextY) + fdY(X[-1], Y[-1])) / 2 * h

        X.append(nextX)
        Y.append(nextY)

        dX = nextdX
        dY = nextdY
    return torch.tensor(X), torch.tensor(Y), times
