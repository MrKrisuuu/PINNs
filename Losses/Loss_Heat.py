from get_points import get_initial_points, get_interior_points, get_boundary_points
from PINN import f, dfdt, dfdx, dfdy
from Losses.Loss import Loss
from constants.initial_conditions import get_initial_conditions
from constants.constants_Heat import get_Heat_start_level, get_Heat_level

import torch


class Loss_Heat(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        dimension = 3
        if len(self.args) != dimension:
            raise Exception(f"This problem is in {dimension}D, not in {len(self.args)}D")

    def residual_loss(self, pinn):
        x, y, t = get_interior_points(*self.args, n_points=self.n_points, device=pinn.device())

        loss = dfdt(pinn, x, y, t) - dfdx(pinn, x, y, t, order=2) - dfdy(pinn, x, y, t, order=2)

        return loss.pow(2).mean()

    def initial_loss(self, pinn):
        x, y, t = get_initial_points(*self.args, n_points=self.n_points, device=pinn.device())

        init = get_initial_conditions("Heat")

        loss = f(pinn, x, y, t) - init(x, y)

        return loss.pow(2).mean()

    def boundary_loss(self, pinn):
        down, up, left, right = get_boundary_points(*self.args, n_points=self.n_points, device=pinn.device())
        x_down,  y_down,  t_down    = down
        x_up,    y_up,    t_up      = up
        x_left,  y_left,  t_left    = left
        x_right, y_right, t_right   = right

        loss_down  = dfdy( pinn, x_down,  y_down,  t_down  )
        loss_up    = dfdy( pinn, x_up,    y_up,    t_up    )
        loss_left  = dfdx( pinn, x_left,  y_left,  t_left  )
        loss_right = dfdx( pinn, x_right, y_right, t_right )

        return loss_down.pow(2).mean()  + \
            loss_up.pow(2).mean()    + \
            loss_left.pow(2).mean()  + \
            loss_right.pow(2).mean()

    def invariant_loss(self, pinn):
        x, y, t = get_initial_points(*self.args, n_points=self.n_points, device=pinn.device())

        s = get_Heat_start_level(x, y)
        t = torch.full_like(x, 0)
        loss = get_Heat_level(f(pinn, x, y, t)).reshape(1)
        for t in torch.linspace(self.args[2][0], self.args[2][1], self.n_points).to(pinn.device())[1:]:
            t = torch.full_like(x, t)
            loss = torch.cat((loss, get_Heat_level(f(pinn, x, y, t)).reshape(1)))

        loss = loss.to(pinn.device())

        return (loss - s).pow(2).mean()
