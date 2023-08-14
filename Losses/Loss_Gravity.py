from get_points import get_boundary_points, get_initial_points, get_interior_points
from PINN import PINN, f, dfdx, dfdy, dfdt
from Losses.Loss import Loss


class Loss_Gravity(Loss):
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

        r = (f(pinn, t, output_value=0)**2 + f(pinn, t, output_value=1)**2)**(1/2)

        # GM = 1
        eq1 = dfdt(pinn, t, order=2, output_value=0) + f(pinn, t, output_value=0) / r**3
        eq2 = dfdt(pinn, t, order=2, output_value=1) + f(pinn, t, output_value=1) / r**3

        return eq1.pow(2).mean() + eq2.pow(2).mean()

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

        e = 0

        cx1 = f(pinn, t, output_value=0) - (1 - e)
        cx2 = dfdt(pinn, t, output_value=0)
        cy1 = f(pinn, t, output_value=1)
        cy2 = dfdt(pinn, t, output_value=1) - ((1+e)/(1-e))**(1/2)

        return cx1.pow(2).mean() + cx2.pow(2).mean() + cy1.pow(2).mean() + cy2.pow(2).mean()

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

        r = (f(pinn, t, output_value=0) ** 2 + f(pinn, t, output_value=1) ** 2) ** (1 / 2)
        energy = (dfdt(pinn, t, output_value=0) ** 2 + dfdt(pinn, t, output_value=1) ** 2) / 2 - 1 / r

        momentum = f(pinn, t, output_value=0) * dfdt(pinn, t, output_value=1) - \
              f(pinn, t, output_value=1) * dfdt(pinn, t, output_value=0)

        help1 = energy - (-0.5)
        help2 = momentum - (1)

        return help1.pow(2).mean() + help2.pow(2).mean()