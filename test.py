import torch

from other_methods_SIR import euler_SIR, semi_SIR, implictit_SIR, RK_SIR
from other_methods_Gravity import euler_Gravity, semi_Gravity, implicit_Gravity, RK_Gravity
from utils import get_derivatives, get_derivatives_from_pinn
from PINN import dfdt
from plotting import print_loss, plot_loss, plot_1D, plot_1D_in_2D, plot_3D, plot_compare, plot_difference
from other_plotting import plot_Gravity_energy, plot_Gravity_momentum, plot_Tsunami_level


def test_SIR(loss, pinn, loss_values, t_domain):
    # Result of training
    print_loss(loss, pinn)
    plot_loss(loss_values, name="loss_SIR", save="loss_SIR")
    t = torch.linspace(t_domain[0], t_domain[1], 101).reshape(-1, 1)
    t.requires_grad = True
    plot_1D(pinn, t, name="SIR", labels=["S", "I", "R"], ylabel="Population", save="SIR_PINN")

    h = 0.001

    # Euler
    S_euler, I_euler, R_euler, times = euler_SIR(t_domain[1])
    v_euler = S_euler + I_euler + R_euler
    dS_euler = get_derivatives(S_euler, h)
    dI_euler = get_derivatives(I_euler, h)
    dR_euler = get_derivatives(R_euler, h)
    d_euler = dS_euler + dI_euler + dR_euler

    # Semi-implicit Euler
    # S_semi, I_semi, R_semi, times = semi_SIR(t_domain[1])
    # v_semi = S_semi + I_semi + R_semi
    # dS_semi = get_derivatives(S_semi, h)
    # dI_semi = get_derivatives(I_semi, h)
    # dR_semi = get_derivatives(R_semi, h)
    # d_semi = dS_semi + dI_semi + dR_semi

    # Implicit Euler
    S_implicit, I_implicit, R_implicit, times = implictit_SIR(t_domain[1])
    v_implicit = S_implicit + I_implicit + R_implicit
    dS_implicit = get_derivatives(S_implicit, h)
    dI_implicit = get_derivatives(I_implicit, h)
    dR_implicit = get_derivatives(R_implicit, h)
    d_implicit = dS_implicit + dI_implicit + dR_implicit

    # RK
    S_RK, I_RK, R_RK, _ = RK_SIR(t_domain[1])
    v_RK = S_RK + I_RK + R_RK
    dS_RK = get_derivatives(S_RK, h)
    dI_RK = get_derivatives(I_RK, h)
    dR_RK = get_derivatives(R_RK, h)
    d_RK = dS_RK + dI_RK + dR_RK

    # PINN
    SIR = pinn(times.reshape(-1, 1)).detach().cpu().numpy()
    S_pinn = torch.tensor(SIR[:, 0])
    I_pinn = torch.tensor(SIR[:, 1])
    R_pinn = torch.tensor(SIR[:, 2])
    v_pinn = S_pinn + I_pinn + R_pinn
    dS_pinn = get_derivatives_from_pinn(pinn, times, dfdt, output_value=0)
    dI_pinn = get_derivatives_from_pinn(pinn, times, dfdt, output_value=1)
    dR_pinn = get_derivatives_from_pinn(pinn, times, dfdt, output_value=2)
    d_pinn = dS_pinn + dI_pinn + dR_pinn

    # Compare methods
    plot_compare([S_euler, S_implicit, S_RK, S_pinn], times, ["Euler", "Implicit", "RK4", "PINN"], name="Susceptible individuals", ylabel="Susceptible individuals", save="S")
    plot_compare([I_euler, I_implicit, I_RK, I_pinn], times, ["Euler", "Implicit", "RK4", "PINN"], name="Infectious individuals", ylabel="Infectious individuals", save="I")
    plot_compare([R_euler, R_implicit, R_RK, R_pinn], times, ["Euler", "Implicit", "RK4", "PINN"], name="Removed individuals", ylabel="Removed individuals", save="R")
    plot_difference([v_euler, v_implicit, v_RK, v_pinn], times, torch.full_like(times, 1), ["Euler", "Implicit", "RK4", "PINN"], name="Difference in total population", ylabel="Difference", save="Total")
    plot_difference([d_euler, d_implicit, d_RK, d_pinn], times, torch.full_like(times, 0), ["Euler", "Implicit", "RK4", "PINN"], name="Difference in change of population", ylabel="Difference", save="Change")


def test_Gravity(loss, pinn, loss_values, t_domain):
    # Results of training
    print_loss(loss, pinn)
    plot_loss(loss_values, name="loss_Gravity")

    t = torch.linspace(t_domain[0], t_domain[1], 101).reshape(-1, 1)
    t.requires_grad = True

    plot_1D_in_2D(pinn, t, name="Orbit")
    plot_1D(pinn, t, name="Gravity (sins)", labels=["X", "Y"], ylabel="Value")

    h = 0.001

    # Euler
    X_euler, Y_euler, times = euler_Gravity(t_domain[1], h)
    dX_euler = get_derivatives(X_euler, h)
    dY_euler = get_derivatives(Y_euler, h)
    r_euler = (X_euler**2 + Y_euler**2)**(1/2)
    energy_euler = (dX_euler**2 + dY_euler**2) / 2 - 1 / r_euler
    momentum_euler = X_euler * dY_euler - Y_euler * dX_euler

    # Semi-implicit Euler
    X_semi, Y_semi, times = semi_Gravity(t_domain[1], h)
    dX_semi = get_derivatives(X_semi, h)
    dY_semi = get_derivatives(Y_semi, h)
    r_semi = (X_semi ** 2 + Y_semi ** 2) ** (1 / 2)
    energy_semi = (dX_semi ** 2 + dY_semi ** 2) / 2 - 1 / r_semi
    momentum_semi = X_semi * dY_semi - Y_semi * dX_semi

    # Implicit Euler
    X_implicit, Y_implicit, times = implicit_Gravity(t_domain[1], h)
    dX_implicit = get_derivatives(X_implicit, h)
    dY_implicit = get_derivatives(Y_implicit, h)
    r_implicit = (X_implicit ** 2 + Y_implicit ** 2) ** (1 / 2)
    energy_implicit = (dX_implicit ** 2 + dY_implicit ** 2) / 2 - 1 / r_implicit
    momentum_implicit = X_implicit * dY_implicit - Y_implicit * dX_implicit

    # RK
    X_RK, Y_RK, _ = RK_Gravity(t_domain[1], h)
    dX_RK = get_derivatives(X_RK, h)
    dY_RK = get_derivatives(Y_RK, h)
    r_RK = (X_RK ** 2 + Y_RK ** 2) ** (1 / 2)
    energy_RK = (dX_RK ** 2 + dY_RK ** 2) / 2 - 1 / r_RK
    momentum_RK = X_RK * dY_RK - Y_RK * dX_RK

    # PINN
    Gravity = pinn(times.reshape(-1, 1)).detach().cpu().numpy()
    X_pinn = torch.tensor(Gravity[:, 0])
    Y_pinn = torch.tensor(Gravity[:, 1])
    dX_pinn = get_derivatives_from_pinn(pinn, times, dfdt, output_value=0)
    dY_pinn = get_derivatives_from_pinn(pinn, times, dfdt, output_value=1)
    r_pinn = (X_pinn**2 + Y_pinn**2)**(1/2)
    energy_pinn = (dX_pinn**2 + dY_pinn**2) / 2 - 1 / r_pinn
    momentum_pinn = X_pinn * dY_pinn - Y_pinn * dX_pinn

    # Compare methods
    plot_compare([X_euler, X_semi, X_implicit, X_RK, X_pinn], times, ["Euler", "Semi", "Implicit", "RK4", "PINN"], name="X coordinate", ylabel="X")
    plot_compare([Y_euler, Y_semi, Y_implicit, Y_RK, Y_pinn], times, ["Euler", "Semi", "Implicit", "RK4", "PINN"], name="Y coordinate", ylabel="Y")
    plot_difference([energy_euler, energy_semi, energy_implicit, energy_RK, energy_pinn], times, torch.full_like(times, -0.5), ["Euler", "Semi", "Implicit", "RK4", "PINN"], name="Difference in energy", ylabel="Difference")
    plot_difference([momentum_euler, momentum_semi, momentum_implicit, momentum_RK, momentum_pinn], times, torch.full_like(times, 1), ["Euler", "Semi", "Implicit", "RK4", "PINN"], name="Difference in momentum", ylabel="Difference")


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