from constants.initial_conditions import get_initial_conditions

from scipy.integrate import quad
import torch
import time

BEGIN = 0
END = 3
G = 1
PI = 3.14


def e(i, x, n):
    if x<BEGIN and x>END:
        return 0
    elif x<(i-1)/n*(END-BEGIN)+BEGIN or x>(i+1)/n*(END-BEGIN)+BEGIN:
        return 0
    elif x<(i)/n*(END-BEGIN)+BEGIN:
        return x-(i-1)/n*(END-BEGIN)+BEGIN
    else:
        return (i+1)/n*(END-BEGIN)+BEGIN-x


def ePrim(i, x, n):
    if x<BEGIN and x>END:
        return 0
    elif x<(i-1)/n*(END-BEGIN)+BEGIN or x>(i+1)/n*(END-BEGIN)+BEGIN:
        return 0
    elif x<(i)/n*(END-BEGIN)+BEGIN:
        return 1
    else:
        return -1


# def integral(f, a, b, n):
#     sum = 0
#     m = 1000*n
#     for i in range(m):
#         sum += (f((i)*(b-a)/(m)+a)+f((i+1)*(b-a)/(m)+a))/2*(b-a)/(m)
#     return sum, 0


def B(i, j, n):
    if abs(i-j) >= 2:
        return 0
    if abs(i - j) == 1:
        return (END-BEGIN)/n
    if abs(i - j) == 0:
        return -2*(END-BEGIN)/n
    result, error = quad(lambda x: -ePrim(i, x, n)*ePrim(j, x, n), 0, 3)
    return result


def L(j, n):
    # if j==0 or j==n:
    #     return 4*PI*G*(END-BEGIN)/n*(END-BEGIN)/n/2
    # if j>0 and j<n:
    #     return 4*PI*G*(END-BEGIN)/n*(END-BEGIN)/n

    x1 = (END-BEGIN)*j/n+BEGIN - (END-BEGIN)/n
    x2 = (END-BEGIN)*j/n+BEGIN + (END-BEGIN)/n
    x1 = max(1, x1)
    x2 = min(2, x2)
    result, error = quad(lambda x: e(j, x, n), x1, x2)

    # result, error = quad(lambda x: e(j, x, n), 1, 2)
    return 4*PI*G*result


def FEM_Poisson(points, n=300):
    start = time.time()
    M = [[None for _ in range(n+2)] for _ in range(n+1)]

    for i in range(n+1):
        for j in range(n + 1):
            M[i][j] = B(i, j, n)

    for j in range(n + 1):
        M[j][n+1] = L(j, n)

    for i in range(n + 2):
        M[0][i] = 0
        M[n][i] = 0

    for i in range(n + 1):
        M[i][0] = 0
        M[i][n] = 0

    M[0][0] = 1
    M[n][n] = 1

    for i in range(n):
        for j in range(i+1, n+1):
            C = M[j][i]/M[i][i]
            for k in range(i, n+2):
                M[j][k] -= C*M[i][k]

    for i in range(n, -1, -1):
        M[i][n+1] /= M[i][i]
        M[i][i] = 1
        for j in range(i-1, -1, -1):
            M[j][n+1] -= M[j][i] * M[i][n+1]
            M[j][i] = 0

    W = [None for _ in range(n+1)]
    for i in range(n+1):
        W[i] = M[i][n+1]

    x = 0
    y = 0
    Xs = []
    Ys = []
    left_shitf, right_shift = get_initial_conditions("Poisson")
    for i in range(points):
        x = i/(points-1)*(END-BEGIN)+BEGIN
        for j in range(n+1):
            y+= W[j]*e(j, x, n)
        y += (right_shift - left_shitf)*x/3+left_shitf
        Xs.append(x)
        Ys.append(y)
        x = 0
        y = 0

    print(f"Time for Poisson: {round(time.time() - start, 3)} ")
    return torch.tensor(Ys), torch.tensor(Xs)
    # p = torch.linspace(0, 3, 1001)
    # return torch.tensor([e(n, x, n) for x in p]), p