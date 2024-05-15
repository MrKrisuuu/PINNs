from get_points import get_initial_points, get_interior_points
from PINN import f, dfdt
from Losses.Loss import Loss
from constants.initial_conditions import get_initial_conditions
from constants.constants_Kepler import get_Kepler_start_energy, get_Kepler_energy, get_Kepler_start_moment, get_Kepler_moment


class Loss_Kepler(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        dimension = 1
        if len(self.args) != dimension:
            raise Exception(f"This problem is in {dimension}D, not in {len(self.args)}D")

    def residual_loss(self, pinn):
        t = get_interior_points(*self.args, n_points=self.n_points, device=pinn.device())

        r = (f(pinn, t, output_value=0)**2 + f(pinn, t, output_value=1)**2)**(1/2)

        # GM = 1
        eq1 = dfdt(pinn, t, order=2, output_value=0) + f(pinn, t, output_value=0) / r**3
        eq2 = dfdt(pinn, t, order=2, output_value=1) + f(pinn, t, output_value=1) / r**3

        return eq1.pow(2).mean() + eq2.pow(2).mean()

    def initial_loss(self, pinn):
        t = get_initial_points(*self.args, n_points=self.n_points, device=pinn.device())

        (X, Y, dX, dY) = get_initial_conditions("Kepler")

        cx1 = f(pinn, t, output_value=0) - X[0]
        cx2 = dfdt(pinn, t, output_value=0) - dX
        cy1 = f(pinn, t, output_value=1) - Y[0]
        cy2 = dfdt(pinn, t, output_value=1) - dY

        return cx1.pow(2).mean() + cx2.pow(2).mean() + cy1.pow(2).mean() + cy2.pow(2).mean()

    def invariant_loss(self, pinn):
        t = get_interior_points(*self.args, n_points=self.n_points, device=pinn.device())

        X = f(pinn, t, output_value=0)
        dX = dfdt(pinn, t, output_value=0)
        Y = f(pinn, t, output_value=1)
        dY = dfdt(pinn, t, output_value=1)

        invariant1 = get_Kepler_energy(X, Y, dX, dY) - get_Kepler_start_energy()
        invariant2 = get_Kepler_moment(X, Y, dX, dY) - get_Kepler_start_moment()

        return invariant1.pow(2).mean() + invariant2.pow(2).mean()