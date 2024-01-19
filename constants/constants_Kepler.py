from constants.initial_conditions import get_initial_conditions


def get_Kepler_start_energy():
    (X, Y, dX, dY) = get_initial_conditions("Kepler")
    R = (X[0] ** 2 + Y[0] ** 2) ** (1 / 2)
    return (dX ** 2 + dY ** 2) / 2 - 1 / R


def get_Kepler_energy(X, Y, dX, dY):
    R = (X ** 2 + Y ** 2) ** (1 / 2)
    return (dX ** 2 + dY ** 2) / 2 - 1 / R


def get_Kepler_start_moment():
    (X, Y, dX, dY) = get_initial_conditions("Kepler")
    return X[0] * dY - Y[0] * dX


def get_Kepler_moment(X, Y, dX, dY):
    return X * dY - Y * dX