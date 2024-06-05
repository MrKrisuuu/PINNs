import torch
import numpy as np

from PINN import PINN
from Losses.Loss_SIR import Loss_SIR
from Losses.Loss_Kepler import Loss_Kepler
from Losses.Loss_LV import Loss_LV
from Losses.Loss_Poisson import Loss_Poisson
from Losses.Loss_Heat import Loss_Heat
from training.train import pretrain_model, train_model, device

from other_methods.other_methods_SIR import euler_SIR
from other_methods.other_methods_Kepler import euler_Kepler
from other_methods.other_methods_Lotka_Volterra import euler_LV
from other_methods.other_methods_Poisson import FEM_Poisson


def train(pinn, loss, epochs=10000, pretrain_epochs=None, LBFGS_epochs=None, pre_func=None, pre_args=None):
    loss_values = []
    adam_epochs = epochs
    if pretrain_epochs:
        adam_epochs -= pretrain_epochs
    if LBFGS_epochs:
        adam_epochs -= LBFGS_epochs

    if pretrain_epochs:
        result_pre = pre_func(*pre_args)
        times = result_pre[-1]
        results = torch.stack(result_pre[:-1], dim=1).to(pinn.device())
        times = times.reshape(-1, 1).to(device)
        pinn, loss_values_pre = pretrain_model(pinn, times, results, loss, epochs=pretrain_epochs)
        loss_values.append(loss_values_pre)

    if adam_epochs:
        pinn, loss_values_adam = train_model(pinn, loss, epochs=adam_epochs, optim=torch.optim.Adam)
        loss_values.append(loss_values_adam)

    if LBFGS_epochs:
        pinn, loss_values_lbfgs = train_model(pinn, loss, epochs=LBFGS_epochs, optim=torch.optim.LBFGS)
        loss_values.append(loss_values_lbfgs)

    # best_pinn = best_pinn.cpu()

    return loss, pinn, loss_values


def train_SIR(t_domain, epochs=1000, pretrain_epochs=None, LBFGS_epochs=None, invariant=False):
    pinn = PINN(1, 3).to(device)

    loss = Loss_SIR(
        t_domain,
        invariant=invariant
    )

    loss, best_pinn, loss_values = train(pinn, loss, epochs, pretrain_epochs, LBFGS_epochs, pre_func=euler_SIR, pre_args=(t_domain[1], ))

    return loss, best_pinn, loss_values


def train_Kepler(t_domain, epochs=50000, pretrain_epochs=None, LBFGS_epochs=None, invariant=False):
    pinn = PINN(1, 2).to(device)

    loss = Loss_Kepler(
        t_domain,
        invariant=invariant
    )

    loss, best_pinn, loss_values = train(pinn, loss, epochs, pretrain_epochs, LBFGS_epochs, pre_func=euler_Kepler, pre_args=(t_domain[1], ))

    return loss, best_pinn, loss_values


def train_LV(t_domain, epochs=20000, pretrain_epochs=None, LBFGS_epochs=None, invariant=False):
    pinn = PINN(1, 2).to(device)

    loss = Loss_LV(
        t_domain,
        invariant=invariant
    )

    loss, best_pinn, loss_values = train(pinn, loss, epochs, pretrain_epochs, LBFGS_epochs, pre_func=euler_LV, pre_args=(t_domain[1], ))

    return loss, best_pinn, loss_values


def train_Poisson(t_domain, epochs=5000, pretrain_epochs=None, LBFGS_epochs=None, invariant=False):
    pinn = PINN(1, 1).to(device)

    loss = Loss_Poisson(
        t_domain,
        invariant=invariant
    )

    loss, best_pinn, loss_values = train(pinn, loss, epochs, pretrain_epochs, LBFGS_epochs, pre_func=FEM_Poisson, pre_args=(301, 10))

    return loss, best_pinn, loss_values


def train_Heat(x_domain_Heat, y_domain_Heat, t_domain_Heat, epochs=10000, pretrain_epochs=None, LBFGS_epochs=None, invariant=False):
    pinn = PINN(3, 1).to(device)

    loss = Loss_Heat(
        x_domain_Heat,
        y_domain_Heat,
        t_domain_Heat,
        invariant=invariant
    )

    loss, best_pinn, loss_values = train(pinn, loss, epochs, pretrain_epochs, LBFGS_epochs, pre_func=None)

    return loss, best_pinn, loss_values
