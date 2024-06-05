import torch
import scipy

from other_methods.other_methods_utils import get_RK_change, get_times
from constants.initial_conditions import get_initial_conditions


def fS(s, i, r, params):
    b, y, N = params
    return - b / N * i * s


def fI(s, i, r, params):
    b, y, N = params
    return b / N * i * s - y * i


def fR(s, i, r, params):
    b, y, N = params
    return y * i


def SIR_equation(values, previous, h, params):
    s_n1, i_n1, r_n1 = values
    s_n, i_n, r_n = previous
    b, y, N = params
    S = s_n1 - s_n + h * b / N * i_n1 * s_n1
    I = i_n1 - i_n - h * b / N * i_n1 * s_n1 + h * y * i_n1
    R = r_n1 - r_n - h * y * i_n1
    return [S, I, R]


def euler_SIR(time, h=0.01):
    S, I, R, params = get_initial_conditions("SIR")
    dS = fS(S[-1], I[-1], R[-1], params)
    dI = fI(S[-1], I[-1], R[-1], params)
    dR = fR(S[-1], I[-1], R[-1], params)
    times = get_times(time, h)
    for t in times[1:]:
        S.append(S[-1] + dS * h)
        I.append(I[-1] + dI * h)
        R.append(R[-1] + dR * h)
        dS = fS(S[-1], I[-1], R[-1], params)
        dI = fI(S[-1], I[-1], R[-1], params)
        dR = fR(S[-1], I[-1], R[-1], params)
    return torch.tensor(S), torch.tensor(I), torch.tensor(R), times


def semi_SIR(time, h=0.01):
    # Is it possible to implement it?
    pass


def implicit_SIR(time, h=0.01):
    S, I, R, params = get_initial_conditions("SIR")
    times = get_times(time, h)
    for t in times[1:]:
        prev = (S[-1], I[-1], R[-1])
        s, i, r = scipy.optimize.fsolve(SIR_equation, prev, args=(prev, h, params))
        S.append(s)
        I.append(i)
        R.append(r)
    return torch.tensor(S), torch.tensor(I), torch.tensor(R), times


def RK_SIR(time, h=0.01):
    S, I, R, params = get_initial_conditions("SIR")
    times = get_times(time, h)
    for t in times[1:]:
        dS = get_RK_change(fS, h, S[-1], I[-1], R[-1], params)
        dI = get_RK_change(fI, h, S[-1], I[-1], R[-1], params)
        dR = get_RK_change(fR, h, S[-1], I[-1], R[-1], params)

        S.append(S[-1] + dS)
        I.append(I[-1] + dI)
        R.append(R[-1] + dR)
    return torch.tensor(S), torch.tensor(I), torch.tensor(R), times