from get_points import get_boundary_points, get_initial_points, get_interior_points
from PINN import PINN, f, dfdx, dfdy, dfdt
from Losses.Loss import Loss
import torch


class Loss_VL(Loss):
    def residual_loss(self, pinn):
        x, y, t = None, None, None
        if len(self.args) == 1:
            t = get_interior_points(*self.args, n_points=self.n_points, device=pinn.device())
        elif len(self.args) == 2:
            x, t = get_interior_points(*self.args, n_points=self.n_points, device=pinn.device())
        elif len(self.args) == 3:
            x, y, t = get_interior_points(*self.args, n_points=self.n_points, device=pinn.device())
        else:
            raise Exception(f"Too many arguments: {len(self.args)}, expected 1, 2 or 3.")

        a, b, c, d = (1, 1, 1, 2)

        prey = dfdt(pinn, t, output_value=0) - (a - b * f(pinn, t, output_value=1)) * f(pinn, t, output_value=0)
        predator = dfdt(pinn, t, output_value=1) - (c * f(pinn, t, output_value=0) - d) * f(pinn, t, output_value=1)

        loss = prey.pow(2) + predator.pow(2)

        return loss.mean()

    def initial_loss(self, pinn):
        x, y, t = None, None, None
        if len(self.args) == 1:
            t = get_initial_points(*self.args, n_points=self.n_points, device=pinn.device())
        elif len(self.args) == 2:
            x, t = get_initial_points(*self.args, n_points=self.n_points, device=pinn.device())
        elif len(self.args) == 3:
            x, y, t = get_initial_points(*self.args, n_points=self.n_points, device=pinn.device())
        else:
            raise Exception(f"Too many arguments: {len(self.args)}, expected 1, 2 or 3.")

        prey = f(pinn, t, output_value=0) - 1
        predtor = f(pinn, t, output_value=1) - 1

        loss = prey.pow(2) + predtor.pow(2)

        return loss.mean()

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

        x = f(pinn, t, output_value=0)
        y = f(pinn, t, output_value=1)

        x = torch.where(x < 0, torch.tensor(0.001), x)
        y = torch.where(y < 0, torch.tensor(0.001), y)

        c = 2 * torch.log(x) - x + torch.log(y) - y

        return (c+2).pow(2).mean()
