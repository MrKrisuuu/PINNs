from get_points import get_initial_points, get_interior_points
from PINN import PINN, f, dfdt
from Losses.Loss import Loss
import torch
from constants.initial_conditions import get_initial_conditions
from constants.constants_LV import get_LV_start_c, get_LV_c


class Loss_LV(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        dimension = 1
        if len(self.args) != dimension:
            raise Exception(f"This problem is in {dimension}D, not in {len(self.args)}D")

    def residual_loss(self, pinn):
        t = get_interior_points(*self.args, n_points=self.n_points, device=pinn.device())

        (_, params) = get_initial_conditions("LV")
        a, b, c, d = params

        prey = dfdt(pinn, t, output_value=0) - (a - b * f(pinn, t, output_value=1)) * f(pinn, t, output_value=0)
        predator = dfdt(pinn, t, output_value=1) - (c * f(pinn, t, output_value=0) - d) * f(pinn, t, output_value=1)

        loss = prey.pow(2) + predator.pow(2)

        return loss.mean()

    def initial_loss(self, pinn):
        t = get_initial_points(*self.args, n_points=self.n_points, device=pinn.device())

        (y, _) = get_initial_conditions("LV")

        prey = f(pinn, t, output_value=0) - y[0]
        predtor = f(pinn, t, output_value=1) - y[1]

        loss = prey.pow(2) + predtor.pow(2)

        return loss.mean()

    def invariant_loss(self, pinn: PINN):
        t = get_interior_points(*self.args, n_points=self.n_points, device=pinn.device())

        X = f(pinn, t, output_value=0)
        Y = f(pinn, t, output_value=1)

        r = torch.cat((X, Y), dim=1)

        return (get_LV_c(r) - get_LV_start_c()).pow(2).mean()
