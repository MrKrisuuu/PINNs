import torch

from PINN import PINN
from Losses.Loss_SIR import Loss_SIR
from Losses.Loss_Gravity import Loss_Gravity
from Losses.Loss_Tsunami import Loss_Tsunami
from train import train_model
from plotting import print_loss, plot_loss, plot_1D, plot_1D_in_2D, plot_3D, plot_compare
from other_plotting import plot_SIR_number, plot_SIR_change, plot_Gravity_energy, plot_Gravity_momentum, plot_Tsunami_level
from torch import nn
from other_methods import euler_SIR, RK_SIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def train_SIR(t_domain):
    pinn = PINN(1, 3).to(device)

    loss = Loss_SIR(
        t_domain,
        n_points=200
    )

    best_pinn, loss_values = train_model(pinn, loss_fn=loss, max_epochs=10)
    best_pinn = best_pinn.cpu()

    torch.save(best_pinn, "./results/SIR.pth")

    return loss, best_pinn, loss_values


def test_SRI(loss, pinn, loss_values, t_domain):
    print_loss(loss, pinn)
    plot_loss(loss_values, name="loss_SIR")

    t = torch.linspace(t_domain[0], t_domain[1], 101).reshape(-1, 1)
    t.requires_grad = True

    plot_1D(pinn, t, name="SIR", labels=["S", "I", "R"], ylabel="Population")
    plot_SIR_number(pinn, t)
    plot_SIR_change(pinn, t)


def train_Gravity(t_domain):
    pinn = PINN(1, 2).to(device)

    loss = Loss_Gravity(
        t_domain,
        n_points=1000,
        help=True
    )

    best_pinn, loss_values = train_model(pinn, loss_fn=loss, max_epochs=1000)
    best_pinn = best_pinn.cpu()

    torch.save(best_pinn, "./results/Gravity.pth")

    return loss, best_pinn, loss_values


def test_Gravity(loss, pinn, loss_values, t_domain):
    print_loss(loss, pinn)
    plot_loss(loss_values, name="loss_Gravity")

    t = torch.linspace(t_domain[0], t_domain[1], 101).reshape(-1, 1)
    t.requires_grad = True

    plot_1D_in_2D(pinn, t, name="Gravity")
    plot_1D(pinn, t, name="Gravity (sins)", labels=["X", "Y"], ylabel="Value")
    plot_Gravity_energy(pinn, t)
    plot_Gravity_momentum(pinn, t)


def train_Tsunami(x_domain, y_domain, t_domain):
    pinn = PINN(3, 1).to(device)

    loss = Loss_Tsunami(
        x_domain,
        y_domain,
        t_domain,
        n_points=15
    )

    best_pinn, loss_values = train_model(pinn, loss_fn=loss, max_epochs=500)
    best_pinn = best_pinn.cpu()

    torch.save(best_pinn, "./results/Tsunami.pth")

    return loss, best_pinn, loss_values


def test_Tsunami(loss, pinn, loss_values, x_domain, y_domain, t_domain):
    print_loss(loss, pinn)
    plot_loss(loss_values, name="loss_Tsunami")

    x = torch.linspace(x_domain[0], x_domain[1], 101).reshape(-1, 1)
    x.requires_grad = True
    y = torch.linspace(y_domain[0], y_domain[1], 101).reshape(-1, 1)
    y.requires_grad = True
    t = torch.linspace(t_domain[0], t_domain[1], 101).reshape(-1, 1)
    t.requires_grad = True

    plot_3D(pinn, x, y, t, name="Tsunami")
    plot_Tsunami_level(pinn, x, y, t)


if __name__ == "__main__":
    t_domain_SIR = [0, 10]

    # t_domain_Gravity = [0, 1]

    # x_domain_Tsunami = [0, 1]
    # y_domain_Tsunami = [0, 1]
    # t_domain_Tsunami = [0, 1]

    loss_SIR, best_pinn_SIR, loss_values_SIR = train_SIR(t_domain_SIR)
    # loss_Gravity, best_pinn_Gravity, loss_values_Gravity = train_Gravity(t_domain_Gravity)
    # loss_Tsunami, best_pinn_Tsunami, loss_values_Tsunami = train_Tsunami(x_domain_Tsunami, y_domain_Tsunami, t_domain_Tsunami)

    # test_SRI(loss_SIR, best_pinn_SIR, loss_values_SIR, t_domain_SIR)
    S_euler, I_euler, R_euler, times = euler_SIR(10)
    S_RK, I_RK, R_RK, times = RK_SIR(10)
    SIR = best_pinn_SIR(torch.tensor(times).reshape(-1, 1)).detach().cpu().numpy()
    S_pinn = SIR[:, 0]
    I_pinn = SIR[:, 1]
    R_pinn = SIR[:, 2]
    plot_compare([S_euler, S_RK, S_pinn], times, ["Euler", "RK4", "PINN"], name="S")
    plot_compare([I_euler, I_RK, I_pinn], times, ["Euler", "RK4", "PINN"], name="I")
    plot_compare([R_euler, R_RK, R_pinn], times, ["Euler", "RK4", "PINN"], name="R")


    # test_Gravity(loss_Gravity, best_pinn_Gravity, loss_values_Gravity, t_domain_Gravity)
    # test_Tsunami(loss_Tsunami, best_pinn_Tsunami, loss_values_Tsunami, x_domain_Tsunami, y_domain_Tsunami, t_domain_Tsunami)