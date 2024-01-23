import numpy as np
from copy import deepcopy
import torch
import matplotlib.pyplot as plt
import imageio


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Only 1D
def save_pinn(pinn, times, epoch):
    plt.xlim(times[0].item(), times[-1].item())
    plt.ylim(-5, 5)
    plt.plot(times.detach().cpu().numpy(), pinn(times).detach().cpu().numpy())
    plt.title(f"Epoch: {epoch}")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.grid(True)
    plt.savefig(f"./gif/{epoch}.png")
    plt.clf()
    return f"./gif/{epoch}.png"


def save_gif(files, name):
    with imageio.get_writer(f"./gif/{name}.gif", mode="I", loop=0) as writer:
        for filename in files:
            image = imageio.v2.imread(filename)
            writer.append_data(image)


def pretrain_model(nn_approximator, input, target, epochs=1_000):
    optimizer = torch.optim.Adam(nn_approximator.parameters())
    loss_values = []
    min_loss = 10000000000
    best_model = deepcopy(nn_approximator)
    for epoch in range(0, epochs + 1):
        output = nn_approximator(input)
        loss = ((output - target) ** 2).mean()

        if (epoch) % 100 == 0:
            print(f"Epoch of pretrain: {epoch} - Loss: {float(loss):>7f}")

        if min_loss > float(loss):
            best_model = deepcopy(nn_approximator)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return best_model, np.array(loss_values)


def train_model(nn_approximator, loss_fn, epochs=1_000, ratio=0.2, make_gif=False):
    optimizer1 = torch.optim.Adam(nn_approximator.parameters())
    optimizer2 = torch.optim.LBFGS(nn_approximator.parameters())
    loss_values = []
    min_loss = 10000000000
    best_model = deepcopy(nn_approximator)
    files = []
    for epoch in range(0, epochs+1):
        loss, residual_loss, initial_loss, boundary_loss, help_loss = loss_fn(nn_approximator)
        if (epoch) % 50 == 0 and make_gif:
            file = save_pinn(best_model, torch.linspace(loss_fn.args[0][0], loss_fn.args[0][1], 101).reshape(-1, 1).to(device), epoch)
            files.append(file)

        if min_loss > float(loss):
            min_loss = float(loss)
            print(f"Epoch: {epoch} - Loss: {float(loss):>7f}")
            best_model = deepcopy(nn_approximator)
        elif (epoch) % 100 == 0:
            print(f"Epoch: {epoch} - Loss: {float(loss):>7f}")

        loss_values.append(
            [loss.item(), residual_loss.item(), initial_loss.item(), boundary_loss.item(), help_loss.item()])

        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
    if make_gif:
        save_gif(files, str(loss_fn.__class__.__name__))
    return best_model, np.array(loss_values)