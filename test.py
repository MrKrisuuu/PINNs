import torch

from other_methods.other_methods_SIR import euler_SIR, implicit_SIR, RK_SIR
from other_methods.other_methods_Kepler import euler_Kepler, semi_Kepler, implicit_Kepler, RK_Kepler, Verlet_Kepler
from other_methods.other_methods_Volterra_Lotka import euler_VL, implicit_VL, RK_VL

from utils import get_derivatives, get_values_from_pinn, get_derivatives_from_pinn
from PINN import dfdt
from plotting import print_loss, plot_loss, plot_1D, plot_1D_in_2D, plot_3D, plot_compare, plot_difference


def test_SIR(loss, pinn, loss_values, t_domain):
    def get_SIR_sum(S, I, R):
        return S + I + R

    # Result of training
    print_loss(loss, pinn)
    plot_loss(loss_values, name="loss_SIR", save="loss_SIR")
    t = torch.linspace(t_domain[0], t_domain[1], 101).reshape(-1, 1)
    t.requires_grad = True
    plot_1D(pinn, t, name="SIR", labels=["S", "I", "R"], ylabel="Population", save="SIR_PINN")

    h = 0.001

    # Euler
    S_euler, I_euler, R_euler, times = euler_SIR(t_domain[1], h)
    v_euler = get_SIR_sum(S_euler, I_euler, R_euler)

    # Semi-implicit Euler
    # S_semi, I_semi, R_semi, times = semi_SIR(t_domain[1], h)
    # v_semi = get_SIR_sum(S_semi, I_semi, R_semi)

    # Implicit Euler
    S_implicit, I_implicit, R_implicit, times = implicit_SIR(t_domain[1], h)
    v_implicit = get_SIR_sum(S_implicit, I_implicit, R_implicit)

    # RK
    S_RK, I_RK, R_RK, times = RK_SIR(t_domain[1], h)
    v_RK = get_SIR_sum(S_RK, I_RK, R_RK)

    # PINN
    S_pinn, I_pinn, R_pinn = get_values_from_pinn(pinn, times)
    v_pinn = get_SIR_sum(S_pinn, I_pinn, R_pinn)

    # Compare methods
    plot_compare([S_euler, S_implicit, S_RK, S_pinn], times, ["Euler", "Implicit", "RK4", "PINN"], name="Susceptible individuals", ylabel="Susceptible individuals", save="S")
    plot_compare([I_euler, I_implicit, I_RK, I_pinn], times, ["Euler", "Implicit", "RK4", "PINN"], name="Infectious individuals", ylabel="Infectious individuals", save="I")
    plot_compare([R_euler, R_implicit, R_RK, R_pinn], times, ["Euler", "Implicit", "RK4", "PINN"], name="Removed individuals", ylabel="Removed individuals", save="R")
    plot_difference([v_euler, v_implicit, v_RK, v_pinn], times, torch.full_like(times, 1), ["Euler", "Implicit", "RK4", "PINN"], name="Difference in total population", ylabel="Difference", save="Total")


def test_Kepler(loss, pinn, loss_values, t_domain):
    def get_Kepler_energy(X, Y, dX, dY):
        R = (X ** 2 + Y ** 2) ** (1 / 2)
        return (dX ** 2 + dY ** 2) / 2 - 1 / R

    def get_Kepler_moment(X, Y, dX, dY):
        return X * dY - Y * dX

    # Results of training
    print_loss(loss, pinn)
    plot_loss(loss_values, name="loss_Kepler")

    t = torch.linspace(t_domain[0], t_domain[1], 101).reshape(-1, 1)
    t.requires_grad = True

    plot_1D_in_2D(pinn, t, name="Orbit")
    plot_1D(pinn, t, name="Kepler (sins)", labels=["X", "Y"], ylabel="Value")

    h = 0.001

    # Euler
    X_euler, Y_euler, times = euler_Kepler(t_domain[1], h)
    dX_euler = get_derivatives(X_euler, h)
    dY_euler = get_derivatives(Y_euler, h)
    energy_euler = get_Kepler_energy(X_euler, Y_euler, dX_euler, dY_euler)
    momentum_euler = get_Kepler_moment(X_euler, Y_euler, dX_euler, dY_euler)

    # Semi-implicit Euler
    X_semi, Y_semi, times = semi_Kepler(t_domain[1], h)
    dX_semi = get_derivatives(X_semi, h)
    dY_semi = get_derivatives(Y_semi, h)
    energy_semi = get_Kepler_energy(X_semi, Y_semi, dX_semi, dY_semi)
    momentum_semi = get_Kepler_moment(X_semi, Y_semi, dX_semi, dY_semi)

    # Implicit Euler
    X_implicit, Y_implicit, times = implicit_Kepler(t_domain[1], h)
    dX_implicit = get_derivatives(X_implicit, h)
    dY_implicit = get_derivatives(Y_implicit, h)
    energy_implicit = get_Kepler_energy(X_implicit, Y_implicit, dX_implicit, dY_implicit)
    momentum_implicit = get_Kepler_moment(X_implicit, Y_implicit, dX_implicit, dY_implicit)

    # RK
    X_RK, Y_RK, _ = RK_Kepler(t_domain[1], h)
    dX_RK = get_derivatives(X_RK, h)
    dY_RK = get_derivatives(Y_RK, h)
    energy_RK = get_Kepler_energy(X_RK, Y_RK, dX_RK, dY_RK)
    momentum_RK = get_Kepler_moment(X_RK, Y_RK, dX_RK, dY_RK)

    # Verlet
    X_Verlet, Y_Verlet, _ = Verlet_Kepler(t_domain[1], h)
    dX_Verlet = get_derivatives(X_Verlet, h)
    dY_Verlet = get_derivatives(Y_Verlet, h)
    energy_Verlet = get_Kepler_energy(X_Verlet, Y_Verlet, dX_Verlet, dY_Verlet)
    momentum_Verlet = get_Kepler_moment(X_Verlet, Y_Verlet, dX_Verlet, dY_Verlet)

    # PINN
    X_pinn, Y_pinn = get_values_from_pinn(pinn, times)
    dX_pinn = get_derivatives_from_pinn(pinn, times, dfdt, output_value=0)
    dY_pinn = get_derivatives_from_pinn(pinn, times, dfdt, output_value=1)
    energy_pinn = get_Kepler_energy(X_pinn, Y_pinn, dX_pinn, dY_pinn)
    momentum_pinn = get_Kepler_moment(X_pinn, Y_pinn, dX_pinn, dY_pinn)

    # Compare methods
    plot_compare([X_euler, X_semi, X_implicit, X_RK, X_Verlet, X_pinn], times, ["Euler", "Semi", "Implicit", "RK4", "Verlet", "PINN"], name="X coordinate", ylabel="X")
    plot_compare([Y_euler, Y_semi, Y_implicit, Y_RK, Y_Verlet, Y_pinn], times, ["Euler", "Semi", "Implicit", "RK4", "Verlet", "PINN"], name="Y coordinate", ylabel="Y")
    plot_difference([energy_euler, energy_semi, energy_implicit, energy_RK, energy_Verlet, energy_pinn], times, torch.full_like(times, -0.5), ["Euler", "Semi", "Implicit", "RK4", "Verlet", "PINN"], name="Difference in energy", ylabel="Difference")
    plot_difference([momentum_euler, momentum_semi, momentum_implicit, momentum_RK, momentum_Verlet, momentum_pinn], times, torch.full_like(times, 1), ["Euler", "Semi", "Implicit", "RK4", "Verlet", "PINN"], name="Difference in momentum", ylabel="Difference")


def test_VL(loss, pinn, loss_values, t_domain, h=0.001):
    def get_VL_c(X, Y):
        return 2 * torch.log(X) - X + torch.log(Y) - Y

    # Result of training
    print_loss(loss, pinn)
    plot_loss(loss_values, name="loss_VL", save="loss_VL")
    t = torch.linspace(t_domain[0], t_domain[1], 101).reshape(-1, 1)
    t.requires_grad = True
    plot_1D(pinn, t, name="VL", labels=["X", "Y"], ylabel="Population", save="VL_PINN")

    # Euler
    X_euler, Y_euler, times = euler_VL(t_domain[1], h)
    c_euler = get_VL_c(X_euler, Y_euler)

    # Semi-implicit Euler
    # X_semi, Y_semi, times = semi_VL(t_domain[1], h)
    # c_semi = get_VL_c(X, Y)

    # Implicit Euler
    X_implicit, Y_implicit, times = implicit_VL(t_domain[1], h)
    c_implicit = get_VL_c(X_implicit, Y_implicit)

    # RK
    X_RK, Y_RK, times = RK_VL(t_domain[1], h)
    c_RK = get_VL_c(X_RK, Y_RK)

    # PINN
    X_pinn, Y_pinn = get_values_from_pinn(pinn, times)
    c_pinn = get_VL_c(X_pinn, Y_pinn)

    # Compare methods
    plot_compare([X_euler, X_implicit, X_RK, X_pinn], times, ["Euler", "Implicit", "RK4", "PINN"], name="Prey individuals", ylabel="Prey individuals", save="Prey")
    plot_compare([Y_euler, Y_implicit, Y_RK, Y_pinn], times, ["Euler", "Implicit", "RK4", "PINN"], name="Predators individuals", ylabel="Predators individuals", save="Predators")
    plot_difference([c_euler, c_implicit, c_RK, c_pinn], times, torch.full_like(times, -2), ["Euler", "Implicit", "RK4", "PINN"], name="Constant in VL", ylabel="Difference", save="Constant")
