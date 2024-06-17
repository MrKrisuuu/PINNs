from constants.initial_conditions import get_initial_conditions


def get_Kepler_start_energy():
    (y, _) = get_initial_conditions("Kepler")
    R = (y[0] ** 2 + y[1] ** 2) ** (1 / 2)
    return (y[2] ** 2 + y[3] ** 2) / 2 - 1 / R


def get_Kepler_energy(y):
    R = (y[:, 0] ** 2 + y[:, 1] ** 2) ** (1 / 2)
    return (y[:, 2] ** 2 + y[:, 3] ** 2) / 2 - 1 / R


def get_Kepler_start_moment():
    (y, _) = get_initial_conditions("Kepler")
    return y[0] * y[3] - y[1] * y[2]


def get_Kepler_moment(y):
    return y[:, 0] * y[:, 3] - y[:, 1] * y[:, 2]