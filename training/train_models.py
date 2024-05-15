import torch
import numpy as np

from PINN import PINN
from Losses.Loss_SIR import Loss_SIR
from Losses.Loss_Kepler import Loss_Kepler
from Losses.Loss_LV import Loss_LV
from training.train import pretrain_model, train_model, device

from other_methods.other_methods_SIR import euler_SIR
from other_methods.other_methods_Kepler import euler_Kepler
from other_methods.other_methods_Lotka_Volterra import euler_LV


def train_SIR(t_domain, epochs=1000, pretrain_epochs=None, invariant=False):
    pinn = PINN(1, 3).to(device)

    loss = Loss_SIR(
        t_domain,
        invariant=invariant
    )

    if pretrain_epochs:
        S, I, R, times = euler_SIR(t_domain[1])
        results = torch.stack((S, I, R), dim=1).to(device)
        times = times.reshape(-1, 1).to(device)
        pinn, loss_values_pre = pretrain_model(pinn, times, results, loss, epochs=pretrain_epochs)

    if pretrain_epochs:
        new_epochs = epochs - pretrain_epochs
    else:
        new_epochs = epochs
    best_pinn, loss_values_train = train_model(pinn, loss, epochs=new_epochs)
    best_pinn = best_pinn.cpu()

    if pretrain_epochs:
        loss_values = (loss_values_pre, loss_values_train)
    else:
        loss_values = loss_values_train

    #torch.save(best_pinn, "../results/SIR.pth")

    return loss, best_pinn, loss_values


def train_Kepler(t_domain, epochs=50000, pretrain_epochs=None, invariant=False):
    pinn = PINN(1, 2).to(device)

    loss = Loss_Kepler(
        t_domain,
        invariant=invariant
    )

    if pretrain_epochs:
        X, Y, times = euler_Kepler(t_domain[1])
        results = torch.stack((X, Y), dim=1).to(device)
        times = times.reshape(-1, 1).to(device)
        pinn, loss_values_pre = pretrain_model(pinn, times, results, loss, epochs=pretrain_epochs)

    if pretrain_epochs:
        new_epochs = epochs - pretrain_epochs
    else:
        new_epochs = epochs
    best_pinn, loss_values_train = train_model(pinn, loss, epochs=new_epochs)
    best_pinn = best_pinn.cpu()

    if pretrain_epochs:
        loss_values = (loss_values_pre, loss_values_train)
    else:
        loss_values = loss_values_train

    #torch.save(best_pinn, "../results/Kepler.pth")

    return loss, best_pinn, loss_values


def train_LV(t_domain, epochs=20000, pretrain_epochs=None, invariant=False):
    pinn = PINN(1, 2).to(device)

    loss = Loss_LV(
        t_domain,
        invariant=invariant
    )

    if pretrain_epochs:
        X, Y, times = euler_LV(t_domain[1])
        results = torch.stack((X, Y), dim=1).to(device)
        times = times.reshape(-1, 1).to(device)
        pinn, loss_values_pre = pretrain_model(pinn, times, results, loss, epochs=pretrain_epochs)

    if pretrain_epochs:
        new_epochs = epochs - pretrain_epochs
    else:
        new_epochs = epochs
    best_pinn, loss_values_train = train_model(pinn, loss, epochs=new_epochs)
    best_pinn = best_pinn.cpu()

    if pretrain_epochs:
        loss_values = (loss_values_pre, loss_values_train)
    else:
        loss_values = loss_values_train

    #torch.save(best_pinn, "../results/LV.pth")

    return loss, best_pinn, loss_values