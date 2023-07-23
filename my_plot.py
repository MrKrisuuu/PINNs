import matplotlib.pyplot as plt
import torch
import imageio


def plot_1D(pinn, t):
    plt.plot(t.detach().cpu().numpy(), pinn(t).detach().cpu().numpy())
    plt.show()


def plot_2D(pinn, x, t):
    files = []
    for t_raw in t:
        t0 = torch.full_like(x, t_raw.item())
        plt.ylim(-2, 2)
        plt.plot(x.detach().cpu().numpy(), pinn(x, t0).detach().cpu().numpy())
        time = round(t_raw.item(), 2)
        plt.title(f"Step: {time}")
        plt.savefig(f"./plot2D/{time}.png")
        files.append(f"./plot2D/{time}.png")
        plt.clf()

    with imageio.get_writer("mygif.gif", mode="I") as writer:
        for filename in files:
            image = imageio.v2.imread(filename)
            writer.append_data(image)
