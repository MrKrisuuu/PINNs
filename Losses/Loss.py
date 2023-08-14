from PINN import PINN, f, dfdx, dfdy, dfdt
from get_points import get_boundary_points, get_initial_points, get_interior_points
import torch


class Loss:
    def __init__(
            self,
            *args: [float, float],

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

        return torch.tensor(0)

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

        return torch.tensor(0)

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

        return torch.tensor(0)

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

        return torch.tensor(0)

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
            loss = Loss(*some_args)
            calculated_loss = loss(pinn)
        ```
        """
        return self.verbose(pinn)
