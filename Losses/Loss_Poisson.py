from get_points import get_initial_points, get_interior_points, get_boundary_points
from PINN import f, dfdt
from Losses.Loss import Loss
from constants.initial_conditions import get_initial_conditions

import torch


class Loss_Poisson(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        dimension = 1
        if len(self.args) != dimension:
            raise Exception(f"This problem is in {dimension}D, not in {len(self.args)}D")

    def residual_loss(self, pinn):
        t = get_interior_points(*self.args, n_points=self.n_points, device=pinn.device())

        loss = dfdt(pinn, t, order=2) - 4*3.14*torch.where(1<torch.where(t<=2, t, 0), 1, 0)

        return loss.pow(2).mean()

    def boundary_loss(self, pinn):
        (t0, t1) = get_boundary_points(*self.args, n_points=self.n_points, device=pinn.device())

        (x1, x2) = get_initial_conditions("Poisson")

        l1 = f(pinn, t0) - x1
        l2 = f(pinn, t1) - x2

        loss = l1.pow(2) + l2.pow(2)

        return loss.mean()
