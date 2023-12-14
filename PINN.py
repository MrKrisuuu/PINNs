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
        if len(args) == 1:
            if self.input == 1:
                t = args[0]
                stack = torch.cat([t], dim=1)
            else:
                raise Exception(f"Wrong numbers of arguments: 1, expected {self.input}.")
        elif len(args) == 2:
            if self.input == 2:
                x = args[0]
                t = args[1]
                stack = torch.cat([x, t], dim=1)
            else:
                raise Exception(f"Wrong numbers of arguments: 2, expected {self.input}.")
        elif len(args) == 3:
            if self.input == 3:
                x = args[0]
                y = args[1]
                t = args[2]
                stack = torch.cat([x, y, t], dim=1)
            else:
                raise Exception(f"Wrong numbers of arguments: 3, expected {self.input}.")
        else:
            raise Exception(f"Too many arguments: {len(args)}, expected 1, 2 or 3.")

        out = self.act(self.layer_in(stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        logits = self.layer_out(out)

        return logits

    def device(self):
        return next(self.parameters()).device


def f(pinn, *args, output_value=0) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""
    if len(args) == 1:
        if pinn.input == 1:
            t = args[0]
            return pinn(t)[:, output_value:output_value+1]
        else:
            raise Exception(f"Wrong numbers of arguments: 1, expected {pinn.input}.")
    elif len(args) == 2:
        if pinn.input == 2:
            x = args[0]
            t = args[1]
            return pinn(x, t)[:, output_value:output_value+1]
        else:
            raise Exception(f"Wrong numbers of arguments: 2, expected {pinn.input}.")
    elif len(args) == 3:
        if pinn.input == 3:
            x = args[0]
            y = args[1]
            t = args[2]
            return pinn(x, y, t)[:, output_value:output_value+1]
        else:
            raise Exception(f"Wrong numbers of arguments: 3, expected {pinn.input}.")
    else:
        raise Exception(f"Too many arguments: {len(args)}, expected 1, 2 or 3.")


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
    if len(args) == 1:
        if pinn.input == 1:
            t = args[0]
            f_value = f(pinn, t, output_value=output_value)
        else:
            raise Exception(f"Wrong numbers of arguments: 1, expected {pinn.input}.")
    elif len(args) == 2:
        if pinn.input == 2:
            x = args[0]
            t = args[1]
            f_value = f(pinn, x, t, output_value=output_value)
        else:
            raise Exception(f"Wrong numbers of arguments: 2, expected {pinn.input}.")
    elif len(args) == 3:
        if pinn.input == 3:
            x = args[0]
            y = args[1]
            t = args[2]
            f_value = f(pinn, x, y, t, output_value=output_value)
        else:
            raise Exception(f"Wrong numbers of arguments: 3, expected {pinn.input}.")
    else:
        raise Exception(f"Wrong numbers of arguments: {len(args)}, expected 1, 2 or 3.")
    return df(f_value, t, order=order)


def dfdx(pinn, *args, order=1, output_value=0):
    if len(args) == 2:
        if pinn.input == 2:
            x = args[0]
            t = args[1]
            f_value = f(pinn, x, t, output_value=output_value)
        else:
            raise Exception(f"Wrong numbers of arguments: 2, expected {pinn.input}.")
    elif len(args) == 3:
        if pinn.input == 3:
            x = args[0]
            y = args[1]
            t = args[2]
            f_value = f(pinn, x, y, t, output_value=output_value)
        else:
            raise Exception(f"Wrong numbers of arguments: 3, expected {pinn.input}.")
    else:
        raise Exception(f"Wrong numbers of arguments: {len(args)}, expected 2 or 3.")
    return df(f_value, x, order=order)


def dfdy(pinn, *args, order=1, output_value=0):
    if len(args) == 3:
        if pinn.input == 3:
            x = args[0]
            y = args[1]
            t = args[2]
            f_value = f(pinn, x, y, t, output_value=output_value)
        else:
            raise Exception(f"Wrong numbers of arguments: 3, expected {pinn.input}.")
    else:
        raise Exception(f"Wrong numbers of arguments: {len(args)}, expected 3.")
    return df(f_value, y, order=order)
