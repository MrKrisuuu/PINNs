from constants.initial_conditions import get_initial_conditions


def get_Heat_start_level(x, y):
    init = get_initial_conditions("Heat")
    return init(x, y).mean()


def get_Heat_level(z):
    return z.mean()