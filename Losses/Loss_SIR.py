from get_points import get_initial_points, get_interior_points
from PINN import PINN, f, dfdt
from Losses.Loss import Loss
from constants.initial_conditions import get_initial_conditions
from constants.constants_SIR import get_SIR_start_sum, get_SIR_sum


class Loss_SIR(Loss):
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

        (_, _, _, params) = get_initial_conditions("SIR")
        (b, y) = params

        S = dfdt(pinn, t, output_value=0) + b * f(pinn, t, output_value=1) * f(pinn, t, output_value=0)
        I = dfdt(pinn, t, output_value=1) - b * f(pinn, t, output_value=1) * f(pinn, t, output_value=0) + y * f(pinn, t, output_value=1)
        R = dfdt(pinn, t, output_value=2) - y * f(pinn, t, output_value=1)

        loss = S.pow(2) + I.pow(2) + R.pow(2)

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

        (S, I, R, _) = get_initial_conditions("SIR")

        S = f(pinn, t, output_value=0) - S[0]
        I = f(pinn, t, output_value=1) - I[0]
        R = f(pinn, t, output_value=2) - R[0]

        loss = S.pow(2) + I.pow(2) + R.pow(2)

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

        S = f(pinn, t, output_value=0)
        I = f(pinn, t, output_value=1)
        R = f(pinn, t, output_value=2)
        val1 = get_SIR_sum(S, I, R)

        loss = (val1 - get_SIR_start_sum()).pow(2)

        return loss.mean()