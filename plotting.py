import matplotlib.pyplot as plt
import torch
import imageio
import numpy as np


def print_loss(loss, pinn):
    losses = loss.verbose(pinn)
    print(f'Total loss: \t{losses[0]:.6f}    ({losses[0]:.3E})')
    print(f'Interior loss: \t{losses[1]:.6f}    ({losses[1]:.3E})')
    print(f'Initial loss: \t{losses[2]:.6f}    ({losses[2]:.3E})')
    print(f'Bondary loss: \t{losses[3]:.6f}    ({losses[3]:.3E})')
    print(f'Help loss: \t{losses[4]:.6f}    ({losses[4]:.3E})')


def running_average(y, window=100):
    cumsum = np.cumsum(np.insert(y, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def plot_loss(loss_values, name="loss", window=100, save=None):
    average_loss_total = running_average(loss_values[:, 0], window=window)
    average_loss_residual = running_average(loss_values[:, 1], window=window)
    average_loss_initial = running_average(loss_values[:, 2], window=window)
    average_loss_boundary = running_average(loss_values[:, 3], window=window)
    average_loss_help = running_average(loss_values[:, 4], window=window)
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
    plt.grid(True)
    if save:
        plt.savefig(f"./results/{save}.png")
    plt.show()


def plot_1D(pinn, t, name="1D", labels=None, ylabel="Values", save=None):
    plt.plot(t.detach().cpu().numpy(), pinn(t).detach().cpu().numpy(), label=labels)
    plt.title(name)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    if labels:
        plt.legend()
    plt.grid(True)
    if save:
        plt.savefig(f"./results/{save}.png")
    plt.show()


def plot_1D_in_2D(pinn, t, name="1D_2D", save=None):
    data = pinn(t).detach().cpu().numpy()
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y)
    plt.title(name)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    if save:
        plt.savefig(f"./results/{save}.png")
    plt.show()


def plot_2D(pinn, x, t, name="2D"):
    files = []
    for t_raw in t:
        t0 = torch.full_like(x, t_raw.item())
        plt.ylim(-2, 2)
        plt.plot(x.detach().cpu().numpy(), pinn(x, t0).detach().cpu().numpy())
        time = round(t_raw.item(), 2)
        plt.title(f"Step: {time}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.savefig(f"./plot2D/{time}.png")
        files.append(f"./plot2D/{time}.png")
        plt.clf()

    with imageio.get_writer(f"./results/{name}.gif", mode="I") as writer:
        for filename in files:
            image = imageio.v2.imread(filename)
            writer.append_data(image)


def plot_3D(pinn, x, y, t, name="3D"):
    files = []
    x_grid, y_grid = torch.meshgrid(x.reshape(-1), y.reshape(-1), indexing="ij")
    for t_raw in t:
        t0 = torch.full_like(x_grid, t_raw.item())
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.axes.set_zlim3d(bottom=1, top=3)
        ax.plot_surface(x_grid.detach().cpu().numpy(), y_grid.detach().cpu().numpy(),
                        pinn(x_grid.reshape(-1, 1), y_grid.reshape(-1, 1), t0.reshape(-1, 1)).detach().cpu().numpy().reshape(x_grid.shape))
        time = round(t_raw.item(), 2)
        plt.title(f"Step: {time}")
        plt.savefig(f"./plot3D/{time}.png")
        files.append(f"./plot3D/{time}.png")
        plt.clf()
        plt.close()

    with imageio.get_writer(f"./results/{name}.gif", mode="I") as writer:
        for filename in files:
            image = imageio.v2.imread(filename)
            writer.append_data(image)


def plot_compare(data, time, labels, name="", ylabel="Values", save=None):
    for points, label in zip(data, labels):
        plt.plot(time, points, label=label)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(name)
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig(f"./results/{save}.png")
    plt.show()


def plot_difference(data, time, true, labels, name="", ylabel="Values", save=None):
    plt.plot([min(time), max(time)], [0, 0], color="black", linewidth=1)
    for points, label in zip(data, labels):
        plt.plot(time, points - true, label=label)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(name)
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig(f"./results/{save}.png")
    plt.show()

