import matplotlib.pyplot as plt


def fS(s, i, r):
    b = 2
    y = 1
    return - b * i * s


def fI(s, i, r):
    b = 2
    y = 1
    return b * i * s - y * i


def fR(s, i, r):
    b = 2
    y = 1
    return y * i


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
    return S, I, R, times


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
    return S, I, R, times