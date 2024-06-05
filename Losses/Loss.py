from PINN import PINN
import torch


class Loss:
    def __init__(
            self,
            *args,

            n_points=1000,
            weight_r=1.0,
            weight_b=1.0,
            weight_i=1.0,
            weight_h=1.0,
            invariant=False,

            **kwargs
    ):
        self.args = args

        self.n_points = round(n_points**(1/len(self.args)))

        self.weight_r = weight_r
        self.weight_b = weight_b
        self.weight_i = weight_i
        self.weight_h = weight_h

        self.invariant = invariant

    def residual_loss(self, pinn):
        return torch.tensor(0).to(pinn.device())

    def initial_loss(self, pinn):
        return torch.tensor(0).to(pinn.device())

    def boundary_loss(self, pinn):
        return torch.tensor(0).to(pinn.device())

    def invariant_loss(self, pinn):
        return torch.tensor(0).to(pinn.device())

    def verbose(self, pinn):
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

        if self.invariant:
            invariant_loss = self.invariant_loss(pinn)
            final_loss += self.weight_h * invariant_loss
        else:
            invariant_loss = torch.tensor(0)

        return final_loss, residual_loss, initial_loss, boundary_loss, invariant_loss

    def __call__(self, pinn: PINN):
        """
        Allows you to use instance of this class as if it was a function:
        ```
            loss = Loss(*some_args)
            calculated_loss = loss(pinn)
        ```
        """
        return self.verbose(pinn)
