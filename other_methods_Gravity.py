import torch
import scipy


def fdX(x, y):
    r = (x**2 + y**2)**(1/2)
    return - x / r**3


def fdY(x, y):
    r = (x**2 + y**2)**(1/2)
    return - y / r**3


def Gravity_equation(values, previous, h):
    x_n1, y_n1, dx_n1, dy_n1 = values
    x_n, y_n, dx_n, dy_n = previous
    X = x_n1 - x_n - h * dx_n1
    Y = y_n1 - y_n - h * dy_n1
    r = (x_n1**2 + y_n1**2)**(1/2)
    dX = dx_n1 - dx_n + h * x_n1 / r**3
    dY = dy_n1 - dy_n + h * y_n1 / r**3
    return [X, Y, dX, dY]


def euler_Gravity(time, h=0.01):
    X = [1]
    Y = [0]
    dX = 0
    dY = 1
    times = [0]
    t = 0
    while t < time:
        ddX = fdX(X[-1], Y[-1])
        ddY = fdY(X[-1], Y[-1])
        X.append(X[-1] + dX * h)
        Y.append(Y[-1] + dY * h)
        dX = dX + ddX * h
        dY = dY + ddY * h
        t += h
        times.append(t)
    return torch.tensor(X), torch.tensor(Y), torch.tensor(times)


def semi_Gravity(time, h=0.01):
    X = [1]
    Y = [0]
    dX = 0
    dY = 1
    times = [0]
    t = 0
    while t < time:
        ddX = fdX(X[-1], Y[-1])
        ddY = fdY(X[-1], Y[-1])
        dX = dX + ddX * h
        dY = dY + ddY * h
        X.append(X[-1] + dX * h)
        Y.append(Y[-1] + dY * h)
        t += h
        times.append(t)
    return torch.tensor(X), torch.tensor(Y), torch.tensor(times)


def implicit_Gravity(time, h=0.01):
    X = [1]
    Y = [0]
    dX = 0
    dY = 1
    times = [0]
    t = 0
    while t < time:
        prev = (X[-1], Y[-1], dX, dY)
        (x, y, dx, dy), xd, _, _ = scipy.optimize.fsolve(Gravity_equation, prev, args=(prev, h), full_output=True)
        X.append(x)
        Y.append(y)
        dX = dx
        dY = dy
        t += h
        times.append(t)
    return torch.tensor(X), torch.tensor(Y), torch.tensor(times)


def RK_Gravity(time, h=0.01):
    X = [1]
    Y = [0]
    dX = 0
    dY = 1
    times = [0]
    t = 0
    while t < time:
        k1X = h * dX
        k1Y = h * dY
        k1dX = h * fdX(X[-1], Y[-1])
        k1dY = h * fdY(X[-1], Y[-1])

        k2X = h * (dX + k1X/2)
        k2Y = h * (dY + k1Y/2)
        k2dX = h * fdX(X[-1] + k1dX/2, Y[-1] + k1dX/2)
        k2dY = h * fdY(X[-1] + k1dY/2, Y[-1] + k1dY/2)

        k3X = h * (dX + k2X/2)
        k3Y = h * (dY + k2Y/2)
        k3dX = h * fdX(X[-1] + k2dX/2, Y[-1] + k2dX/2)
        k3dY = h * fdY(X[-1] + k2dY/2, Y[-1] + k2dY/2)

        k4X = h * (dX + k3X)
        k4Y = h * (dY + k3Y)
        k4dX = h * fdX(X[-1] + k3dX, Y[-1] + k3dX)
        k4dY = h * fdY(X[-1] + k3dY, Y[-1] + k3dY)

        d_X = (k1X + 2*k2X + 2*k3X + k4X) / 6
        d_Y = (k1Y + 2*k2Y + 2*k3Y + k4Y) / 6
        d_dX = (k1dX + 2*k2dX + 2*k3dX + k4dX) / 6
        d_dY = (k1dY + 2*k2dY + 2*k3dY + k4dY) / 6

        X.append(X[-1] + d_X)
        Y.append(Y[-1] + d_Y)
        dX = dX + d_dX
        dY = dY + d_dY
        t += h
        times.append(t)
    return torch.tensor(X), torch.tensor(Y), torch.tensor(times)