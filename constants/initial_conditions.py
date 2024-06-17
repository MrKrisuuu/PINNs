import torch


def get_initial_conditions(problem):
    if problem == "SIR":
        S = 0.9
        I = 0.1
        R = 0.0
        params = (2, 1, S+I+R)
        return (torch.tensor([S, I, R]), params)
    elif problem == "Kepler":
        X = 1
        Y = 0
        dX = 0
        dY = 1
        # GM = 1
        params = None
        return (torch.tensor([X, Y, dX, dY]), params)
    elif problem == "LV":
        X = 1
        Y = 1
        params = (1, 1, 1, 2)
        return (torch.tensor([X, Y]), params)
    elif problem == "Poisson":
        x1 = 5
        x2 = 4
        return (x1, x2)
    elif problem == "Heat":
        def init(x, y):
            dx = x - 0.5
            dy = y - 0.5
            r2 = torch.where(8 * (dx ** 2 + dy ** 2) < 1, 8 * (dx ** 2 + dy ** 2), 1)
            return (r2 - 1) ** 2 * (r2 + 1) ** 2
        return init
    else:
        raise Exception("Wrong problem!")