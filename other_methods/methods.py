import torch
import scipy
import numpy as np
import time


def euler(f, t, y0, params=()):
    n = len(t)
    y = torch.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n-1):
        h = t[i+1] - t[i]
        y[i+1] = y[i] + h * f(t, y[i], params)
    return y


def implicit_euler(f, t, y0, params=()):
    n = len(t)
    y = torch.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n-1):
        h = t[i+1] - t[i]
        y_i1 = scipy.optimize.fsolve(lambda v: v - np.array((y[i] + h * f(t, v, params))), y[i])
        y[i+1] = torch.tensor(y_i1)
    return y


def rk4(f, t, y0, params=()):
    start = time.time()
    n = len(t)
    y = torch.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n-1):
        h = t[i+1] - t[i]
        k1 = f(t[i], y[i], params)
        k2 = f(t[i] + h / 2, y[i] + h * k1 / 2, params)
        k3 = f(t[i] + h / 2, y[i] + h * k2 / 2, params)
        k4 = f(t[i] + h, y[i] + h * k3, params)
        y[i + 1] = y[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    print(f"Time for RK4 for {f.__name__}: {round(time.time() - start, 3)} ")
    return y


def semi_euler(f, t, y0, params=()):
    n = len(t)
    half = len(y0) // 2
    y = torch.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n-1):
        h = t[i+1] - t[i]
        y[i+1, half:] = y[i, half:] + h * f(t[i], y[i], params)[half:]
        y[i+1, :half] = y[i, :half] + h * f(t[i], torch.cat((y[i, :half], y[i+1, half:])), params)[:half]
    return y


def verlet(f, t, y0, params=()):
    n = len(t)
    half = len(y0) // 2
    y = torch.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n-1):
        h = t[i+1] - t[i]
        y[i+1, :half] = y[i, :half] + h * y[i, half:] + h * h * f(t[i], y[i], params)[half:] / 2
        y[i+1, half:] = y[i, half:] + h * (f(t[i], torch.cat((y[i+1, :half], y[i, half:])), params)[half:] + f(t[i], y[i], params)[half:]) / 2
    return y
