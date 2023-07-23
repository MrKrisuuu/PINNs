from PINN import PINN, f, dfdx, dfdy, dfdt
from typing import Tuple
from get_points import get_boundary_points, get_initial_points, get_interior_points
import torch


class Loss:
    def __init__(
            self,
            *args: Tuple[float, float],

            n_points: int,
            weight_r: float = 1.0,
            weight_b: float = 1.0,
            weight_i: float = 1.0,
            weight_h: float = 1.0,

            help: bool = False
    ):
        self.args = args

        self.n_points = n_points

        self.weight_r = weight_r
        self.weight_b = weight_b
        self.weight_i = weight_i
        self.weight_h = weight_h

        self.help = help

    def residual_loss(self, pinn: PINN):
        x, y, t = None, None, None
        if len(self.args) == 1:
            t = get_interior_points(*self.args, n_points=self.n_points, device=pinn.device())
        elif len(self.args) == 2:
            x, t = get_interior_points(*self.args, n_points=self.n_points, device=pinn.device())
        elif len(self.args) == 3:
            x, y, t = get_interior_points(*self.args, n_points=self.n_points, device=pinn.device())
        else:
            raise Exception(f"Too many arguments: {len(self.args)}, expected 1, 2 or 3.")

        #loss = dfdt(pinn, x, y, t) - dfdx(pinn, x, y, t, order=2) - dfdy(pinn, x, y, t, order=2)
        # ZMIANA
        # g = 9.81
        # dx1 = dfdx(pinn, x, y, t) * dfdx(pinn, x, y, t)  # bez dna
        # dx2 = f(pinn, x, y, t) * dfdx(pinn, x, y, t, order=2)
        # dy1 = dfdy(pinn, x, y, t) * dfdy(pinn, x, y, t)  # bez dna
        # dy2 = f(pinn, x, y, t) * dfdy(pinn, x, y, t, order=2)
        # loss = dfdt(pinn, x, y, t, order=2) - g * (dx1 + dx2 + dy1 + dy2)

        S = dfdt(pinn, t, output_value=0) + f(pinn, t, output_value=0)
        I = dfdt(pinn, t, output_value=1) - f(pinn, t, output_value=1) * f(pinn, t, output_value=0) + f(pinn, t, output_value=1)
        R = dfdt(pinn, t, output_value=2) - f(pinn, t, output_value=1)

        loss = S.pow(2) + I.pow(2) + R.pow(2)
        # return torch.tensor(0)
        return loss.mean()

    def initial_condition(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # r = torch.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        # res = (r - 0.25) * (r < 0.25)
        # res = (torch.sqrt((x-0.5)**2 + (y-0.5)**2) - 0.25)**2 *  # (torch.sqrt((x-0.5)**2 + (y-0.5)**2) - 0.25)**2
        # res = torch.exp(-(r * 7) ** 2) / 2
        # ZMIANA
        # res = 2 + 2 * torch.exp(-((2*r)**2))
        # res = 2 + 2 * x

        res = 100
        return res

    def initial_loss(self, pinn: PINN):
        x, y, t = None, None, None
        if len(self.args) == 1:
            t = get_initial_points(*self.args, n_points=self.n_points, device=pinn.device())
        elif len(self.args) == 2:
            x, t = get_initial_points(*self.args, n_points=self.n_points, device=pinn.device())
        elif len(self.args) == 3:
            x, y, t = get_initial_points(*self.args, n_points=self.n_points, device=pinn.device())
        else:
            raise Exception(f"Too many arguments: {len(self.args)}, expected 1, 2 or 3.")

        #pinn_init = self.initial_condition(x, y)

        S = f(pinn, t, output_value=0) - 0.99
        I = f(pinn, t, output_value=1) - 0.01
        R = f(pinn, t, output_value=2) - 0

        loss = S.pow(2) + I.pow(2) + R.pow(2)

        return loss.mean()

    def boundary_loss(self, pinn: PINN):
        down, up, left, right, t = None, None, None, None, None
        if len(self.args) == 1:
            (t,) = get_boundary_points(*self.args, n_points=self.n_points, device=pinn.device())
        elif len(self.args) == 2:
            down, up = get_boundary_points(*self.args, n_points=self.n_points, device=pinn.device())
        elif len(self.args) == 3:
            down, up, left, right = get_boundary_points(*self.args, n_points=self.n_points, device=pinn.device())
        else:
            raise Exception(f"Too many arguments: {len(self.args)}, expected 1, 2 or 3.")

        # x_down, y_down, t_down = down
        # x_up, y_up, t_up = up
        # x_left, y_left, t_left = left
        # x_right, y_right, t_right = right
        #
        # loss_down = dfdy(pinn, x_down, y_down, t_down)
        # loss_up = dfdy(pinn, x_up, y_up, t_up)
        # loss_left = dfdx(pinn, x_left, y_left, t_left)
        # loss_right = dfdx(pinn, x_right, y_right, t_right)
        #
        # return loss_down.pow(2).mean() + \
        #        loss_up.pow(2).mean() + \
        #        loss_left.pow(2).mean() + \
        #        loss_right.pow(2).mean()

        return torch.tensor(0)

        return loss.pow(2).mean()

    def help_loss(self, pinn: PINN):
        x, y, t = None, None, None
        if len(self.args) == 1:
            t = get_interior_points(*self.args, n_points=self.n_points, device=pinn.device())
        elif len(self.args) == 2:
            x, t = get_interior_points(*self.args, n_points=self.n_points, device=pinn.device())
        elif len(self.args) == 3:
            x, y, t = get_interior_points(*self.args, n_points=self.n_points, device=pinn.device())
        else:
            raise Exception(f"Too many arguments: {len(self.args)}, expected 1, 2 or 3.")

        S = f(pinn, t, output_value=0)
        I = f(pinn, t, output_value=1)
        R = f(pinn, t, output_value=2)

        loss1 = S + I + R - 1

        S = dfdt(pinn, t, output_value=0)
        I = dfdt(pinn, t, output_value=1)
        R = dfdt(pinn, t, output_value=2)

        loss2 = S + I + R

        return loss1.pow(2).mean() + loss2.pow(2).mean()

    def verbose(self, pinn: PINN):
        """
        Returns all parts of the loss function

        Not used during training! Only for checking the results later.
        """
        residual_loss = self.residual_loss(pinn)
        initial_loss = self.initial_loss(pinn)
        boundary_loss = self.boundary_loss(pinn)

        final_loss = \
            self.weight_r * residual_loss + \
            self.weight_i * initial_loss + \
            self.weight_b * boundary_loss

        if self.help:
            help_loss = self.help_loss(pinn)
            final_loss += self.weight_h * help_loss
        else:
            help_loss = torch.tensor(0)

        return final_loss, residual_loss, initial_loss, boundary_loss, help_loss

    def __call__(self, pinn: PINN):
        """
        Allows you to use instance of this class as if it was a function:

        ```
            >>> loss = Loss(*some_args)
            >>> calculated_loss = loss(pinn)
        ```
        """
        return self.verbose(pinn)

