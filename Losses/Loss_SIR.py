from get_points import get_initial_points, get_interior_points
from PINN import PINN, f, dfdt
from Losses.Loss import Loss
from constants.initial_conditions import get_initial_conditions
from constants.constants_SIR import get_SIR_start_sum, get_SIR_sum


class Loss_SIR(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        dimension = 1
        if len(self.args) != dimension:
            raise Exception(f"This problem is in {dimension}D, not in {len(self.args)}D")

    def residual_loss(self, pinn):
        t = get_interior_points(*self.args, n_points=self.n_points, device=pinn.device())

        (_, params) = get_initial_conditions("SIR")
        (b, g, N) = params

        S = dfdt(pinn, t, output_value=0) + b / N * f(pinn, t, output_value=1) * f(pinn, t, output_value=0)
        I = dfdt(pinn, t, output_value=1) - b / N * f(pinn, t, output_value=1) * f(pinn, t, output_value=0) + g * f(pinn, t, output_value=1)
        R = dfdt(pinn, t, output_value=2) - g * f(pinn, t, output_value=1)

        loss = S.pow(2) + I.pow(2) + R.pow(2)

        return loss.mean()

    def initial_loss(self, pinn):
        t = get_initial_points(*self.args, n_points=self.n_points, device=pinn.device())

        (y, _) = get_initial_conditions("SIR")

        S = f(pinn, t, output_value=0) - y[0]
        I = f(pinn, t, output_value=1) - y[1]
        R = f(pinn, t, output_value=2) - y[2]

        loss = S.pow(2) + I.pow(2) + R.pow(2)

        return loss.mean()

    def invariant_loss(self, pinn: PINN):
        t = get_interior_points(*self.args, n_points=self.n_points, device=pinn.device())

        S = f(pinn, t, output_value=0)
        I = f(pinn, t, output_value=1)
        R = f(pinn, t, output_value=2)
        val1 = get_SIR_sum(S, I, R)

        loss = (val1 - get_SIR_start_sum()).pow(2)

        return loss.mean()