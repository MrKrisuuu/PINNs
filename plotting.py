import matplotlib.pyplot as plt
import numpy as np


def print_loss(loss, pinn):
    losses = loss.verbose(pinn)
    print(f'Total loss: \t{losses[0]:.6f}    ({losses[0]:.3E})')
    print(f'Interior loss: \t{losses[1]:.6f}    ({losses[1]:.3E})')
    print(f'Initial loss: \t{losses[2]:.6f}    ({losses[2]:.3E})')
    print(f'Bondary loss: \t{losses[3]:.6f}    ({losses[3]:.3E})')
    print(f'Invariant loss: \t{losses[4]:.6f}    ({losses[4]:.3E})')


def running_average(values, window=100):
    s = int(window / 2)
    avgs = []
    p = max(0, -s)
    q = min(len(values), s + 1)
    current_sum = sum(values[p:q])
    for i in range(0, len(values)):
        new_p = max(0, i - s)
        new_q = min(len(values), i + s + 1)
        if new_p != p:
            current_sum -= values[p]
        if new_q != q:
            current_sum += values[new_q - 1]
        avgs.append(current_sum / (new_q - new_p + 1))
        p = new_p
        q = new_q
    return np.array(avgs)


def plot_loss(loss_values, window=100, title="problem", save="loss"):
    lines = []
    epochs = []
    average_loss_total = running_average(loss_values[0][:, 0], window=window)
    average_loss_residual = running_average(loss_values[0][:, 1], window=window)
    average_loss_initial = running_average(loss_values[0][:, 2], window=window)
    average_loss_boundary = running_average(loss_values[0][:, 3], window=window)
    average_loss_invariant = running_average(loss_values[0][:, 4], window=window)
    epochs += list(range(len(average_loss_total)))
    for loss in loss_values[1:]:
        lines.append(len(average_loss_total)-1)
        average_loss_total = np.concatenate((average_loss_total, running_average(loss[:, 0], window=window)))
        average_loss_residual = np.concatenate((average_loss_residual, running_average(loss[:, 1], window=window)))
        average_loss_initial = np.concatenate((average_loss_initial, running_average(loss[:, 2], window=window)))
        average_loss_boundary = np.concatenate((average_loss_boundary, running_average(loss[:, 3], window=window)))
        average_loss_invariant = np.concatenate((average_loss_invariant, running_average(loss[:, 4], window=window)))
        epochs += list(range(len(epochs)-1, len(average_loss_total)-1))

    max_height = np.max(average_loss_total)
    min_height = max_height
    if not np.all(average_loss_total == 0):
        plt.plot(epochs, average_loss_total, label="Total loss")
    if not np.all(average_loss_residual == 0):
        plt.plot(epochs, average_loss_residual, label="Residual loss")
        min_height = min(min_height, np.min(average_loss_residual))
    if not np.all(average_loss_initial == 0):
        plt.plot(epochs, average_loss_initial, label="Initial loss")
        min_height = min(min_height, np.min(average_loss_initial))
    if not np.all(average_loss_boundary == 0):
        plt.plot(epochs, average_loss_boundary, label="Boundary loss")
        min_height = min(min_height, np.min(average_loss_boundary))
    if not np.all(average_loss_invariant == 0):
        plt.plot(epochs, average_loss_invariant, label="Invariant loss")
        min_height = min(min_height, np.min(average_loss_invariant))

    for line in lines:
        plt.plot([line, line], [10*max_height, min_height/10], color="black", linewidth=2)

    plt.title(f"Loss function for {title} (running average)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss value")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./results/{save}.png")
    plt.show()


def plot_compare_loss(losses, names, save="loss", window=100):
    for loss, name in zip(losses, names):
        plt.plot(running_average(loss[:, 0], window=window), range(len(loss[:, 0] + 1)), label=name)

    plt.title("Loss function of (running average)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss value")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./results/{save}.png")
    plt.show()


def plot_1D(pinn, t, xlabel="Time", ylabel="Values", labels=None, title="problem", save="plot"):
    plt.plot(t.detach().cpu().numpy(), pinn(t).detach().cpu().numpy(), label=labels)
    plt.title(f"Result for {title} by PINN" )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if labels:
        plt.legend()
    plt.grid(True)
    plt.savefig(f"./results/{save}.png")
    plt.show()


def plot_1D_in_2D(pinn, t, title="1D_2D", save="1D_2D"):
    data = pinn(t).detach().cpu().numpy()
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.savefig(f"./results/{save}.png")
    plt.show()


def plot_compare(data, time, labels, title="", xlabel="Time", ylabel="Values", save="compare"):
    for points, label in zip(data, labels):
        plt.plot(time, points, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./results/{save}.png")
    plt.show()


def plot_difference(data, time, true, labels, title="", ylabel="Values", save="compare"):
    plt.plot([min(time), max(time)], [0, 0], color="black", linewidth=1)
    for points, label in zip(data, labels):
        plt.plot(time, points - true, label=label)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./results/{save}.png")
    plt.show()
