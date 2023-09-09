import torch


def get_derivatives(values, h):
    derivatives = [(-values[2] + 3*values[1] - 2*values[0]) / h]
    for i in range(1, len(values)):
        derivatives.append((values[i] - values[i-1]) / h)
    return torch.tensor(derivatives)


def get_derivatives_from_pinn(pinn, times, derivative, order=1, output_value=0):
    t = times.reshape(-1, 1)
    t.requires_grad = True
    return torch.tensor(derivative(pinn, t, order=order, output_value=output_value).detach().cpu().numpy()).reshape(-1)