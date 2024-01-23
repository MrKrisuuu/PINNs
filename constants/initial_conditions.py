def get_initial_conditions(problem):
    if problem == "SIR":
        S = [0.9]
        I = [0.1]
        R = [0.0]
        params = (2, 1)
        return (S, I, R, params)
    elif problem == "Kepler":
        X = [1]
        Y = [0]
        dX = 0
        dY = 1
        # GM = 1
        return (X, Y, dX, dY)
    elif problem == "LV":
        X = [1]
        Y = [1]
        params = (1, 1, 1, 2)
        return (X, Y, params)
    else:
        raise Exception("Wrong problem!")