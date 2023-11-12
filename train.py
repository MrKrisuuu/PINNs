import numpy as np
from copy import deepcopy
from PINN import PINN
from Losses.Loss_SIR import Loss_SIR
from Losses.Loss_Gravity import Loss_Gravity
from Losses.Loss_Tsunami import Loss_Tsunami
import torch


def train_model(nn_approximator, loss_fn, learning_rate=0.002, max_epochs=1_000):
    optimizer = torch.optim.Adam(nn_approximator.parameters(), lr=learning_rate)
    loss_values = []
    min_loss = 10000000000
    best_model = deepcopy(nn_approximator)
    for epoch in range(max_epochs+1):
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

    return best_model, np.array(loss_values)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def train_SIR(t_domain, epochs=1000, help=False):
    pinn = PINN(1, 3).to(device)

    loss = Loss_SIR(
        t_domain,
        n_points=1000,
        help=help
    )

    best_pinn, loss_values = train_model(pinn, loss_fn=loss, max_epochs=epochs)
    best_pinn = best_pinn.cpu()

    torch.save(best_pinn, "./results/SIR.pth")

    return loss, best_pinn, loss_values


def train_Gravity(t_domain, epochs=5000):
    pinn = PINN(1, 2, dim_hidden=200).to(device)

    loss = Loss_Gravity(
        t_domain,
        n_points=1000,
        help=True
    )

    best_pinn, loss_values = train_model(pinn, loss_fn=loss, max_epochs=epochs)
    best_pinn = best_pinn.cpu()

    torch.save(best_pinn, "./results/Gravity.pth")

    return loss, best_pinn, loss_values


def train_Tsunami(x_domain, y_domain, t_domain, epochs=10000):
    pinn = PINN(3, 1).to(device)

    loss = Loss_Tsunami(
        x_domain,
        y_domain,
        t_domain,
        n_points=10,
        # help=True
    )

    best_pinn, loss_values = train_model(pinn, loss_fn=loss, max_epochs=epochs)
    best_pinn = best_pinn.cpu()

    torch.save(best_pinn, "./results/Tsunami.pth")

    return loss, best_pinn, loss_values