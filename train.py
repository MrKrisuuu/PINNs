from typing import Callable
import numpy as np
from PINN import PINN
import torch
from copy import deepcopy


def train_model(
        nn_approximator: PINN,
        loss_fn: Callable,
        learning_rate: int = 0.002,
        max_epochs: int = 1_000
) -> PINN:
    optimizer = torch.optim.Adam(nn_approximator.parameters(), lr=learning_rate)
    loss_values = []
    min_loss = 10000000000
    best_model = deepcopy(nn_approximator)
    for epoch in range(max_epochs+1):

        try:

            loss, residual_loss, initial_loss, boundary_loss, help_loss = loss_fn(nn_approximator)
            if (epoch) % 100 == 0:
                print(f"Epoch: {epoch} - Loss: {float(loss):>7f}")

            if min_loss > float(loss):
                min_loss = float(loss)
                print(f"Epoch: {epoch} - Loss: {float(loss):>7f}")
                best_model = deepcopy(nn_approximator)

            loss_values.append(
                [loss.item(), residual_loss.item(), initial_loss.item(), boundary_loss.item(), help_loss.item()])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        except KeyboardInterrupt:
            break

    return best_model, np.array(loss_values)