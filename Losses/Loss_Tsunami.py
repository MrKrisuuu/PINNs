from get_points import get_boundary_points, get_initial_points, get_interior_points
from PINN import PINN, f, dfdx, dfdy, dfdt
from Losses.Loss import Loss


class Loss_Tsunami(Loss):
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

        g = 9.81
        dx1 = dfdx(pinn, x, y, t) * dfdx(pinn, x, y, t)  # bez dna
        dx2 = f(pinn, x, y, t) * dfdx(pinn, x, y, t, order=2)
        dy1 = dfdy(pinn, x, y, t) * dfdy(pinn, x, y, t)  # bez dna
        dy2 = f(pinn, x, y, t) * dfdy(pinn, x, y, t, order=2)
        loss = dfdt(pinn, x, y, t, order=2) - g * (dx1 + dx2 + dy1 + dy2)

        return loss.pow(2).mean()

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

        loss = f(pinn, x, y, t) - (1 + 2 * x)

        return loss.pow(2).mean()

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

        x_down, y_down, t_down = down
        x_up, y_up, t_up = up
        x_left, y_left, t_left = left
        x_right, y_right, t_right = right

        loss_down = dfdy(pinn, x_down, y_down, t_down)
        loss_up = dfdy(pinn, x_up, y_up, t_up)
        loss_left = dfdx(pinn, x_left, y_left, t_left)
        loss_right = dfdx(pinn, x_right, y_right, t_right)

        return loss_down.pow(2).mean() + \
               loss_up.pow(2).mean() + \
               loss_left.pow(2).mean() + \
               loss_right.pow(2).mean()