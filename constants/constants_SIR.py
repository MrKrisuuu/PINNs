from constants.initial_conditions import get_initial_conditions


def get_SIR_start_sum():
    (S, I, R, params) = get_initial_conditions("SIR")
    return S[0] + I[0] + R[0]


def get_SIR_sum(S, I, R):
    return S + I + R