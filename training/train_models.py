import torch
import numpy as np
import time

from PINN import PINN
from Losses.Loss_SIR import Loss_SIR
from Losses.Loss_Kepler import Loss_Kepler
from Losses.Loss_LV import Loss_LV
from Losses.Loss_Poisson import Loss_Poisson
from Losses.Loss_Heat import Loss_Heat
from training.train import pretrain_model, train_model, device
from utils import get_times
from constants.initial_conditions import get_initial_conditions

from other_methods.methods import euler
from other_methods.other_methods_Poisson import FEM_Poisson
from other_methods.equations import SIR, Kepler, LV


def train(pinn, loss, epochs=10000, pretrain_epochs=None, LBFGS_epochs=None, pre_func=None, pre_args=None, half=False):
    loss_values = []
    adam_epochs = epochs
    if pretrain_epochs:
        adam_epochs -= pretrain_epochs
    if LBFGS_epochs:
        adam_epochs -= LBFGS_epochs

    if pretrain_epochs:
        result = pre_func(*pre_args)
        if half:
            result = result[:, :len(result[0])//2]
        (_, times, _, _) = pre_args
        pinn, loss_values_pre = pretrain_model(pinn, times.reshape(-1, 1).to(pinn.device()), result.to(pinn.device()), loss, epochs=pretrain_epochs)
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
    start = time.time()

    pinn = PINN(1, 3).to(device)

    loss = Loss_SIR(
        t_domain,
        invariant=invariant
    )

    times = get_times(t_domain[1], 0.01)
    (y0, params) = get_initial_conditions("SIR")
    loss, best_pinn, loss_values = train(pinn, loss, epochs, pretrain_epochs, LBFGS_epochs, pre_func=euler, pre_args=(SIR, times, y0, params))

    print(f"Time for PINN for SIR: {round(time.time() - start, 3)} ")

    return loss, best_pinn, loss_values


def train_Kepler(t_domain, epochs=50000, pretrain_epochs=None, LBFGS_epochs=None, invariant=False):
    start = time.time()

    pinn = PINN(1, 2).to(device)

    loss = Loss_Kepler(
        t_domain,
        invariant=invariant
    )

    times = get_times(t_domain[1], 0.01)
    (y0, params) = get_initial_conditions("Kepler")
    loss, best_pinn, loss_values = train(pinn, loss, epochs, pretrain_epochs, LBFGS_epochs, pre_func=euler, pre_args=(Kepler, times, y0, params), half=True)

    print(f"Time for PINN for Kepler: {round(time.time() - start, 3)} ")

    return loss, best_pinn, loss_values


def train_LV(t_domain, epochs=20000, pretrain_epochs=None, LBFGS_epochs=None, invariant=False):
    start = time.time()

    pinn = PINN(1, 2).to(device)

    loss = Loss_LV(
        t_domain,
        invariant=invariant
    )

    times = get_times(t_domain[1], 0.01)
    (y0, params) = get_initial_conditions("LV")
    loss, best_pinn, loss_values = train(pinn, loss, epochs, pretrain_epochs, LBFGS_epochs, pre_func=euler, pre_args=(LV, times, y0, params))

    print(f"Time for PINN for LV: {round(time.time() - start, 3)} ")

    return loss, best_pinn, loss_values


def train_Poisson(t_domain, epochs=5000, pretrain_epochs=None, LBFGS_epochs=None, invariant=False):
    start = time.time()

    pinn = PINN(1, 1).to(device)

    loss = Loss_Poisson(
        t_domain,
        invariant=invariant
    )

    loss, best_pinn, loss_values = train(pinn, loss, epochs, pretrain_epochs, LBFGS_epochs, pre_func=FEM_Poisson, pre_args=(301, 10))

    print(f"Time for PINN for Poisson: {round(time.time() - start, 3)} ")

    return loss, best_pinn, loss_values


def train_Heat(x_domain_Heat, y_domain_Heat, t_domain_Heat, epochs=50000, pretrain_epochs=None, LBFGS_epochs=None, invariant=False):
    start = time.time()

    pinn = PINN(3, 1).to(device)

    loss = Loss_Heat(
        x_domain_Heat,
        y_domain_Heat,
        t_domain_Heat,
        invariant=invariant
    )

    loss, best_pinn, loss_values = train(pinn, loss, epochs, pretrain_epochs, LBFGS_epochs, pre_func=None)

    print(f"Time for PINN for Heat: {round(time.time() - start, 3)} ")

    return loss, best_pinn, loss_values
