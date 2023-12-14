import torch

from PINN import PINN
from Losses.Loss_SIR import Loss_SIR
from Losses.Loss_Kepler import Loss_Kepler
from Losses.Loss_VL import Loss_VL
from train import pretrain_model, train_model, device

from other_methods.other_methods_SIR import euler_SIR
from other_methods.other_methods_Kepler import euler_Kepler
from other_methods.other_methods_Volterra_Lotka import euler_VL
from train import device


def train_SIR(t_domain, epochs=1000, pretrain_epochs=1000, help=False, make_gif=False):
    pinn = PINN(1, 3).to(device)

    loss = Loss_SIR(
        t_domain,
        n_points=1000,
        help=help
    )

    best_pinn, loss_values = train_model(pinn, loss_fn=loss, epochs=epochs, make_gif=make_gif)
    best_pinn = best_pinn.cpu()

    torch.save(best_pinn, "./results/SIR.pth")

    return loss, best_pinn, loss_values


def train_Kepler(t_domain, epochs=20000, pretrain_epochs=5000, help=False, make_gif=False):
    pinn = PINN(1, 2).to(device)

    loss = Loss_Kepler(
        t_domain,
        n_points=1000,
        help=help
    )

    # X, Y, times = euler_Kepler(t_domain[1])
    # results = torch.stack((X, Y), dim=1).to(device)
    # times = times.reshape(-1, 1).to(device)
    # pinn, loss_values = pretrain_model(pinn, times, results, epochs=pretrain_epochs)

    best_pinn, loss_values = train_model(pinn, loss_fn=loss, epochs=epochs, make_gif=make_gif)
    best_pinn = best_pinn.cpu()

    torch.save(best_pinn, "./results/Kepler.pth")

    return loss, best_pinn, loss_values


def train_VL(t_domain, epochs=20000, pretrain_epochs=5000, help=False, make_gif=False):
    pinn = PINN(1, 2).to(device)

    loss = Loss_VL(
        t_domain,
        n_points=1000,
        help=help
    )

    # X, Y, times = euler_VL(t_domain[1])
    # results = torch.stack((X, Y), dim=1).to(device)
    # times = times.reshape(-1, 1).to(device)
    # pinn, loss_values = pretrain_model(pinn, times, results, epochs=pretrain_epochs)

    best_pinn, loss_values = train_model(pinn, loss_fn=loss, epochs=epochs, make_gif=make_gif)
    best_pinn = best_pinn.cpu()

    torch.save(best_pinn, "./results/VL.pth")

    return loss, best_pinn, loss_values