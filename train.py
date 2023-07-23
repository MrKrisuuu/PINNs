from typing import Callable
import numpy as np
from PINN import PINN
import torch


def train_model(
        nn_approximator: PINN,
        loss_fn: Callable,
        learning_rate: int = 0.01,
        max_epochs: int = 1_000
) -> PINN:
    optimizer = torch.optim.Adam(nn_approximator.parameters(), lr=learning_rate)
    loss_values = []
    min_loss = 10000000000
    for epoch in range(max_epochs):

        try:

            loss, residual_loss, initial_loss, boundary_loss, help_loss = loss_fn(nn_approximator)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_values.append([loss.item(), residual_loss.item(), initial_loss.item(), boundary_loss.item(), help_loss.item()])
            if (epoch + 1) % 100 == 0:
                print(f"Epoch: {epoch + 1} - Loss: {float(loss):>7f}")
            if min_loss > float(loss):
                min_loss = float(loss)
                print(f"Epoch: {epoch + 1} - Loss: {float(loss):>7f}")

        except KeyboardInterrupt:
            break

    return nn_approximator, np.array(loss_values)


def running_average(y, window=100):
    cumsum = np.cumsum(np.insert(y, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)