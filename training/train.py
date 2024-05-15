import numpy as np
from copy import deepcopy
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def pretrain_model(nn_approximator, input, target, loss_fn, epochs=1_000):
    optimizer = torch.optim.Adam(nn_approximator.parameters())
    loss_values = []
    min_loss = 10000000000
    best_model = deepcopy(nn_approximator)
    for epoch in range(0, epochs + 1):
        output = nn_approximator(input)
        loss_pre = ((output - target) ** 2).mean()

        loss, residual_loss, initial_loss, boundary_loss, help_loss = loss_fn(nn_approximator)
        loss_values.append(
            [loss.item(), residual_loss.item(), initial_loss.item(), boundary_loss.item(), help_loss.item()])

        if min_loss > float(loss):
            min_loss = float(loss)
            print(f"Epoch of pretrain: {epoch} - Loss: {float(loss):>7f}")
            best_model = deepcopy(nn_approximator)
        elif (epoch) % 100 == 0:
            print(f"Epoch of pretrain: {epoch} - Loss: {float(loss):>7f}")

        if min_loss > float(loss):
            best_model = deepcopy(nn_approximator)

        optimizer.zero_grad()
        loss_pre.backward()
        optimizer.step()
    return best_model, np.array(loss_values)


def train_model(nn_approximator, loss_fn, epochs=1_000):
    optimizer = torch.optim.Adam(nn_approximator.parameters())
    #optimizer = torch.optim.LBFGS(nn_approximator.parameters())
    loss_values = []
    min_loss = 10000000000
    best_model = deepcopy(nn_approximator)
    for epoch in range(0, epochs):
        loss, residual_loss, initial_loss, boundary_loss, help_loss = loss_fn(nn_approximator)
        loss_values.append(
            [loss.item(), residual_loss.item(), initial_loss.item(), boundary_loss.item(), help_loss.item()])

        if min_loss > float(loss):
            min_loss = float(loss)
            print(f"Epoch: {epoch} - Loss: {float(loss):>7f}")
            best_model = deepcopy(nn_approximator)
        elif (epoch) % 100 == 0:
            print(f"Epoch: {epoch} - Loss: {float(loss):>7f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step(lambda: loss)

    loss, residual_loss, initial_loss, boundary_loss, help_loss = loss_fn(nn_approximator)
    loss_values.append(
        [loss.item(), residual_loss.item(), initial_loss.item(), boundary_loss.item(), help_loss.item()])

    return best_model, np.array(loss_values)