from get_points import get_boundary_points, get_initial_points, get_interior_points
from PINN import PINN, f, dfdx, dfdy, dfdt
from Losses.Loss import Loss


class Loss_SIR(Loss):
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

        b = 2
        y = 1

        S = dfdt(pinn, t, output_value=0) + b * f(pinn, t, output_value=1) * f(pinn, t, output_value=0)
        I = dfdt(pinn, t, output_value=1) - b * f(pinn, t, output_value=1) * f(pinn, t, output_value=0) + y * f(pinn, t, output_value=1)
        R = dfdt(pinn, t, output_value=2) - y * f(pinn, t, output_value=1)

        loss = S.pow(2) + I.pow(2) + R.pow(2)

        return loss.mean()

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

        S = f(pinn, t, output_value=0) - 0.90
        I = f(pinn, t, output_value=1) - 0.10
        R = f(pinn, t, output_value=2) - 0

        loss = S.pow(2) + I.pow(2) + R.pow(2)

        return loss.mean()