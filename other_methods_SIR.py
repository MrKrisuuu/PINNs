import torch
import scipy


def fS(s, i, r, params=(2, 1)):
    b, y = params
    return - b * i * s


def fI(s, i, r, params=(2, 1)):
    b, y = params
    return b * i * s - y * i


def fR(s, i, r, params=(2, 1)):
    b, y = params
    return y * i


def SIR_equation(values, previous, h, params=(2, 1)):
    s_n1, i_n1, r_n1 = values
    s_n, i_n, r_n = previous
    b, y = params
    S = s_n1 - s_n + h * b * i_n1 * s_n1
    I = i_n1 - i_n - h * b * i_n1 * s_n1 + h * y * i_n1
    R = r_n1 - r_n - h * y * i_n1
    return [S, I, R]


def euler_SIR(time, h=0.01):
    S = [0.9]
    I = [0.1]
    R = [0.0]
    times = [0]
    t = 0
    while t < time:
        t += h
        dS = fS(S[-1], I[-1], R[-1])
        dI = fI(S[-1], I[-1], R[-1])
        dR = fR(S[-1], I[-1], R[-1])
        S.append(S[-1] + dS * h)
        I.append(I[-1] + dI * h)
        R.append(R[-1] + dR * h)
        times.append(t)
    return torch.tensor(S), torch.tensor(I), torch.tensor(R), torch.tensor(times)


def implictit_SIR(time, h=0.01):
    S = [0.9]
    I = [0.1]
    R = [0.0]
    times = [0]
    t = 0
    while t < time:
        t += h
        prev = (S[-1], I[-1], R[-1])
        s, i, r = scipy.optimize.fsolve(SIR_equation, prev, args=(prev, h))
        S.append(s)
        I.append(i)
        R.append(r)
        times.append(t)
    return torch.tensor(S), torch.tensor(I), torch.tensor(R), torch.tensor(times)


def RK_SIR(time, h=0.01):
    S = [0.9]
    I = [0.1]
    R = [0.0]
    times = [0]
    t = 0
    while t < time:
        t += h

        k1S = h * fS(S[-1], I[-1], R[-1])
        k1I = h * fI(S[-1], I[-1], R[-1])
        k1R = h * fR(S[-1], I[-1], R[-1])

        k2S = h * fS(S[-1] + k1S/2, I[-1] + k1S/2, R[-1] + k1S/2)
        k2I = h * fI(S[-1] + k1I/2, I[-1] + k1I/2, R[-1] + k1I/2)
        k2R = h * fR(S[-1] + k1R/2, I[-1] + k1R/2, R[-1] + k1R/2)

        k3S = h * fS(S[-1] + k2S/2, I[-1] + k2S/2, R[-1] + k2S/2)
        k3I = h * fI(S[-1] + k2I/2, I[-1] + k2I/2, R[-1] + k2I/2)
        k3R = h * fR(S[-1] + k2R/2, I[-1] + k2R/2, R[-1] + k2R/2)

        k4S = h * fS(S[-1] + k3S, I[-1] + k3S, R[-1] + k3S)
        k4I = h * fI(S[-1] + k3I, I[-1] + k3I, R[-1] + k3I)
        k4R = h * fR(S[-1] + k3R, I[-1] + k3R, R[-1] + k3R)

        dS = (k1S + 2*k2S + 2*k3S + k4S) / 6
        dI = (k1I + 2*k2I + 2*k3I + k4I) / 6
        dR = (k1R + 2*k2R + 2*k3R + k4R) / 6

        S.append(S[-1] + dS)
        I.append(I[-1] + dI)
        R.append(R[-1] + dR)
        times.append(t)
    return torch.tensor(S), torch.tensor(I), torch.tensor(R), torch.tensor(times)