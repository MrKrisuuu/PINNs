import torch


def get_times(time, h):
    return torch.arange(0, time+h, h).to(torch.float32)


def get_derivatives(values, h):
    derivatives = torch.zeros_like(values)
    derivatives[0] = (-values[2, :] + 3*values[1, :] - 2*values[0, :]) / h
    for i in range(1, len(values) - 1):
        derivatives[i] = (values[i+1, :] - values[i-1, :]) / (2*h)
    derivatives[-1] = 2*derivatives[-1, :] - derivatives[-2, :]
    return derivatives


def get_values_from_pinn(pinn, times):
    return pinn(times.reshape(-1, 1).to(pinn.device())).detach().cpu()


def get_derivatives_from_pinn(pinn, times, derivative, order=1, output_value=0):
    t = times.reshape(-1, 1).to(pinn.device())
    t.requires_grad = True
    return torch.tensor(derivative(pinn, t, order=order, output_value=output_value).detach().cpu().numpy()).reshape(-1)