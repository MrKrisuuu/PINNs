import torch


def get_derivatives(values, h):
    derivatives = [(-values[2] + 3*values[1] - 2*values[0]) / h]
    for i in range(1, len(values) - 1):
        derivatives.append((values[i+1] - values[i-1]) / (2*h))
    derivatives.append(2*derivatives[-1] - derivatives[-2])
    return torch.tensor(derivatives)


def get_values_from_pinn(pinn, times):
    values = pinn(times.reshape(-1, 1).to(pinn.device())).detach().cpu().numpy()
    results = []
    for i in range(len(values[0])):
        value = torch.tensor(values[:, i])
        results.append(value)
    return tuple(results)


def get_derivatives_from_pinn(pinn, times, derivative, order=1, output_value=0):
    t = times.reshape(-1, 1)
    t.requires_grad = True
    return torch.tensor(derivative(pinn, t, order=order, output_value=output_value).detach().cpu().numpy()).reshape(-1)