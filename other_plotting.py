import matplotlib.pyplot as plt
import torch
import imageio
import numpy as np


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
        ax.axes.set_xlim3d(0, 1)
        ax.axes.set_ylim3d(0, 1)
        ax.axes.set_zlim3d(0, 1)
        ax.plot_surface(x_grid.detach().cpu().numpy(), y_grid.detach().cpu().numpy(),
                        pinn(x_grid.reshape(-1, 1), y_grid.reshape(-1, 1), t0.reshape(-1, 1)).detach().cpu().numpy().reshape(x_grid.shape))
        time = round(t_raw.item(), 3)
        plt.title(f"Step PINN: {time}")
        plt.savefig(f"./plot3D/{time}.png")
        files.append(f"./plot3D/{time}.png")
        plt.clf()
        plt.close()

    with imageio.get_writer(f"./results/{name}.gif", mode="I") as writer:
        for filename in files:
            image = imageio.v2.imread(filename)
            writer.append_data(image)


def plot_3D_IGA(values, name="3D"):
    files = []
    for (t0, x, y, z) in values:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.axes.set_xlim3d(0, 1)
        ax.axes.set_ylim3d(0, 1)
        ax.axes.set_zlim3d(0, 1)
        N = int(round(len(x)**0.5))
        ax.plot_surface(x.reshape(N, N), y.reshape(N, N), z.reshape(N, N))
        time = round(t0, 3)
        plt.title(f"Step IGA-ADS: {time}")
        plt.savefig(f"./plot3D_IGA/{time}.png")
        files.append(f"./plot3D_IGA/{time}.png")
        plt.clf()
        plt.close()

    with imageio.get_writer(f"./results/{name}.gif", mode="I") as writer:
        for filename in files:
            image = imageio.v2.imread(filename)
            writer.append_data(image)