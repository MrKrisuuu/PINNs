import torch
from torch import nn

class PINN(nn.Module):
    """Simple neural network accepting two features as input and returning a single output

    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """

    def __init__(self, input, output, num_hidden=3, dim_hidden=200, act=nn.Tanh()):
        super().__init__()

        self.input = input
        self.output = output

        self.layer_in = nn.Linear(self.input, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, self.output)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act

    def forward(self, *args: torch.Tensor):
        if len(args) != self.input:
            raise Exception(f"Wrong numbers of dimentions: {len(args)} and {self.input}.")

        if len(args) > 3:
            raise Exception(f"Too many arguments: {len(args)}, expected 1, 2 or 3.")

        stack = torch.cat([*args], dim=1)

        out = self.act(self.layer_in(stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        logits = self.layer_out(out)

        return logits

    def device(self):
        return next(self.parameters()).device


def f(pinn, *args, output_value=0) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""
    if len(args) != pinn.input:
        raise Exception(f"Wrong numbers of dimentions: {len(args)} and {pinn.input}.")

    if len(args) > 3:
        raise Exception(f"Too many arguments: {len(args)}, expected 1, 2 or 3.")

    return pinn(*args)[:, output_value:output_value+1]


def df(output, input, order=1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    df_value = output
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input,
            grad_outputs=torch.ones_like(input),
            create_graph=True,
            retain_graph=True,
        )[0]
    return df_value


def dfdt(pinn, *args, order=1, output_value=0):
    if len(args) != pinn.input:
        raise Exception(f"Wrong numbers of dimentions: {len(args)} and {pinn.input}.")

    if len(args) > 3:
        raise Exception(f"Too many arguments: {len(args)}, expected 1, 2 or 3.")

    f_value = f(pinn, *args, output_value=output_value)

    # t OR x, t OR x, y, t
    return df(f_value, args[-1], order=order)


def dfdx(pinn, *args, order=1, output_value=0):
    if len(args) != pinn.input:
        raise Exception(f"Wrong numbers of dimentions: {len(args)} and {pinn.input}.")

    if len(args) > 3:
        raise Exception(f"Too many arguments: {len(args)}, expected 1, 2 or 3.")

    f_value = f(pinn, *args, output_value=output_value)

    # t OR x, t OR x, y, t
    return df(f_value, args[0], order=order)


def dfdy(pinn, *args, order=1, output_value=0):
    if len(args) != pinn.input:
        raise Exception(f"Wrong numbers of dimentions: {len(args)} and {pinn.input}.")

    if len(args) > 3:
        raise Exception(f"Too many arguments: {len(args)}, expected 1, 2 or 3.")

    f_value = f(pinn, *args, output_value=output_value)

    # t OR x, t OR x, y, t
    return df(f_value, args[1], order=order)
