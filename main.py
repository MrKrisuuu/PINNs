## Pawel Maczuga and Maciej Paszynski 2023


import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

from PINN import PINN, dfdt
from Loss import Loss
from get_points import get_initial_points, get_interior_points, get_boundary_points
from plotting import plot_3D, plot_color
from train import train_model, running_average
from my_plot import plot_1D, plot_2D


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


LENGTH = 10.  # Domain size in x axis. Always starts at 0
TOTAL_TIME = 10.  # Domain size in t axis. Always starts at 0
N_POINTS = 1000  # Number of in single asxis
N_POINTS_PLOT = 150  # Number of points in single axis used in plotting
WEIGHT_RESIDUAL = 1.0  # Weight of residual part of loss function
WEIGHT_INITIAL = 1.0  # Weight of initial part of loss function
WEIGHT_BOUNDARY = 1.0  # Weight of boundary part of loss function
WEIGHT_HELP = 1.0  # Weight of help part of loss function
HELP = True
LAYERS = 4
NEURONS_PER_LAYER = 150
# ZMIANA
EPOCHS = 1000
LEARNING_RATE = 0.002


pinn = PINN(1, 3, LAYERS, NEURONS_PER_LAYER, act=nn.Tanh()).to(device)

x_domain = [0.0, LENGTH]
y_domain = [0.0, LENGTH]
t_domain = [0.0, TOTAL_TIME]

# train the PINN
loss_fn = Loss(
    # x_domain,
    # y_domain,
    t_domain,
    n_points=N_POINTS,
    weight_r=WEIGHT_RESIDUAL,
    weight_b=WEIGHT_INITIAL,
    weight_i=WEIGHT_BOUNDARY,
    weight_h=WEIGHT_HELP,
    help=HELP
)

pinn_trained, loss_values = train_model(
    pinn, loss_fn=loss_fn, learning_rate=LEARNING_RATE, max_epochs=EPOCHS)

pinn = pinn.cpu()

losses = loss_fn.verbose(pinn)
print(f'Total loss: \t{losses[0]:.5f}    ({losses[0]:.3E})')
print(f'Interior loss: \t{losses[1]:.5f}    ({losses[1]:.3E})')
print(f'Initial loss: \t{losses[2]:.5f}    ({losses[2]:.3E})')
print(f'Bondary loss: \t{losses[3]:.5f}    ({losses[3]:.3E})')


# Loss function
average_loss_total = running_average(loss_values[:, 0], window=10)
average_loss_residual = running_average(loss_values[:, 1], window=10)
average_loss_initial = running_average(loss_values[:, 2], window=10)
average_loss_boundary = running_average(loss_values[:, 3], window=10)
average_loss_help = running_average(loss_values[:, 4], window=10)
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.set_title("Loss function (runnig average)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.plot(average_loss_total, label="Total loss")
ax.plot(average_loss_residual, label="Residual loss")
ax.plot(average_loss_initial, label="Initial loss")
ax.plot(average_loss_boundary, label="Boundary loss")
ax.plot(average_loss_help, label="Help loss")
ax.set_yscale('log')
plt.legend()
plt.show()

# Initial condition


x = torch.linspace(x_domain[0], x_domain[1], 101).reshape(-1, 1)
y = torch.linspace(y_domain[0], y_domain[1], 101).reshape(-1, 1)
t = torch.linspace(t_domain[0], t_domain[1], 101).reshape(-1, 1)

plot_1D(pinn, t)


