import torch

from other_methods.other_methods_SIR import euler_SIR, implicit_SIR, RK_SIR
from other_methods.other_methods_Kepler import euler_Kepler, semi_Kepler, implicit_Kepler, RK_Kepler, Verlet_Kepler
from other_methods.other_methods_Lotka_Volterra import euler_LV, implicit_LV, RK_LV
from other_methods.other_methods_Poisson import FEM_Poisson
from other_methods.other_methods_IGA import get_data

from utils import get_derivatives, get_values_from_pinn, get_derivatives_from_pinn
from PINN import dfdt
from plotting import print_loss, plot_loss, plot_1D, plot_1D_in_2D, plot_compare, plot_difference
from other_plotting import plot_3D, plot_3D_IGA

from constants.constants_SIR import get_SIR_start_sum, get_SIR_sum, get_SIR_start_constant, get_SIR_constant
from constants.constants_Kepler import get_Kepler_start_energy, get_Kepler_energy, get_Kepler_start_moment, get_Kepler_moment
from constants.constants_LV import get_LV_start_c, get_LV_c
from constants.constants_Heat import get_Heat_start_level, get_Heat_level


def test_SIR(loss, pinn, loss_values, t_domain, h=0.001, mod=["", ""]):
    # Result of training
    print_loss(loss, pinn)
    plot_loss(loss_values, title="SIR model", save="SIR_loss")
    t = torch.linspace(t_domain[0], t_domain[1], 1001).reshape(-1, 1).to(pinn.device())
    t.requires_grad = True
    plot_1D(pinn, t, title="SIR model", save="SIR_pinn_result", labels=["S", "I", "R"], ylabel="Population")

    # Euler
    S_euler, I_euler, R_euler, times = euler_SIR(t_domain[1], h)
    v_euler = get_SIR_sum(S_euler, I_euler, R_euler)
    c_euler = get_SIR_constant(S_euler, I_euler, R_euler)

    # Implicit Euler
    S_implicit, I_implicit, R_implicit, times = implicit_SIR(t_domain[1], h)
    v_implicit = get_SIR_sum(S_implicit, I_implicit, R_implicit)
    c_implicit = get_SIR_constant(S_implicit, I_implicit, R_implicit)

    # RK
    S_RK, I_RK, R_RK, times = RK_SIR(t_domain[1], h)
    v_RK = get_SIR_sum(S_RK, I_RK, R_RK)
    c_RK = get_SIR_constant(S_RK, I_RK, R_RK)

    # PINN
    S_pinn, I_pinn, R_pinn = get_values_from_pinn(pinn, times)
    v_pinn = get_SIR_sum(S_pinn, I_pinn, R_pinn)
    c_pinn = get_SIR_constant(S_pinn, I_pinn, R_pinn)

    # Compare methods
    plot_compare([S_euler, S_implicit, S_RK, S_pinn], times, ["Euler", "Implicit", "RK4", "PINN"], title="Susceptible individuals", ylabel="Susceptible individuals", save="SIR_S")
    plot_compare([I_euler, I_implicit, I_RK, I_pinn], times, ["Euler", "Implicit", "RK4", "PINN"], title="Infectious individuals", ylabel="Infectious individuals", save="SIR_I")
    plot_compare([R_euler, R_implicit, R_RK, R_pinn], times, ["Euler", "Implicit", "RK4", "PINN"], title="Removed individuals", ylabel="Removed individuals", save="SIR_R")
    plot_difference([v_euler, v_implicit, v_RK, v_pinn], times, torch.full_like(times, get_SIR_start_sum()), ["Euler", "Implicit", "RK4", "PINN"], title="Difference in total population", ylabel="Difference", save="SIR_total")
    plot_difference([c_euler, c_implicit, c_RK, c_pinn], times, torch.full_like(times, get_SIR_start_constant()), ["Euler", "Implicit", "RK4", "PINN"], title="Difference in constant", ylabel="Difference", save="SIR_constant")


def test_Kepler(loss, pinn, loss_values, t_domain, h=0.001, mod=["", ""]):
    # Results of training
    print_loss(loss, pinn)
    plot_loss(loss_values, title=f"Kepler’s problem{mod[0]}", save=f"Kepler{mod[1]}_loss")
    t = torch.linspace(t_domain[0], t_domain[1], 1001).reshape(-1, 1).to(pinn.device())
    t.requires_grad = True
    plot_1D_in_2D(pinn, t, title=f"Orbit{mod[0]}", save=f"Kepler{mod[1]}_pinn_orbit")
    plot_1D(pinn, t, labels=["X", "Y"], ylabel="Value", title="Kepler (sins)", save=f"Kepler{mod[1]}_pinn_result")

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
    plot_compare([X_euler, X_semi, X_implicit, X_RK, X_Verlet, X_pinn], times, ["Euler", "Semi", "Implicit", "RK4", "Verlet", "PINN"], title=f"X coordinate{mod[0]}", ylabel="X", save=f"Kepler{mod[1]}_X")
    plot_compare([Y_euler, Y_semi, Y_implicit, Y_RK, Y_Verlet, Y_pinn], times, ["Euler", "Semi", "Implicit", "RK4", "Verlet", "PINN"], title=f"Y coordinate{mod[0]}", ylabel="Y", save=f"Kepler{mod[1]}_Y")
    plot_difference([energy_euler, energy_semi, energy_implicit, energy_RK, energy_Verlet, energy_pinn], times, get_Kepler_start_energy(), ["Euler", "Semi", "Implicit", "RK4", "Verlet", "PINN"], title=f"Difference in energy{mod[0]}", ylabel="Difference", save=f"Kepler{mod[1]}_energy")
    plot_difference([momentum_euler, momentum_semi, momentum_implicit, momentum_RK, momentum_Verlet, momentum_pinn], times, get_Kepler_start_moment(), ["Euler", "Semi", "Implicit", "RK4", "Verlet", "PINN"], title=f"Difference in momentum{mod[0]}", ylabel="Difference", save=f"Kepler{mod[1]}_momentum")


def test_LV(loss, pinn, loss_values, t_domain, h=0.001, mod=["", ""]):
    # Result of training
    print_loss(loss, pinn)
    plot_loss(loss_values, title=f"Volterra-Lotka problem{mod[0]}", save=f"LV{mod[1]}_loss")
    t = torch.linspace(t_domain[0], t_domain[1], 1001).reshape(-1, 1).to(pinn.device())
    t.requires_grad = True
    plot_1D(pinn, t, title=f"Volterra-Lotka problem{mod[0]}", labels=["X", "Y"], ylabel="Population", save=f"LV{mod[1]}_pinn_result")

    # Euler
    X_euler, Y_euler, times = euler_LV(t_domain[1], h)
    c_euler = get_LV_c(X_euler, Y_euler)

    # Semi-implicit Euler
    # X_semi, Y_semi, times = semi_LV(t_domain[1], h)
    # c_semi = get_LV_c(X, Y)

    # Implicit Euler
    X_implicit, Y_implicit, times = implicit_LV(t_domain[1], h)
    c_implicit = get_LV_c(X_implicit, Y_implicit)

    # RK
    X_RK, Y_RK, times = RK_LV(t_domain[1], h)
    c_RK = get_LV_c(X_RK, Y_RK)

    # PINN
    X_pinn, Y_pinn = get_values_from_pinn(pinn, times)
    c_pinn = get_LV_c(X_pinn, Y_pinn)

    # Compare methods
    plot_compare([X_euler, X_implicit, X_RK, X_pinn], times, ["Euler", "Implicit", "RK4", "PINN"], title="Prey individuals", ylabel="Prey individuals", save=f"LV{mod[1]}_prey")
    plot_compare([Y_euler, Y_implicit, Y_RK, Y_pinn], times, ["Euler", "Implicit", "RK4", "PINN"], title="Predators individuals", ylabel="Predators individuals", save=f"LV{mod[1]}_predator")
    plot_difference([c_euler, c_implicit, c_RK, c_pinn], times, get_LV_start_c(), ["Euler", "Implicit", "RK4", "PINN"], title="Constant in LV", ylabel="Difference", save=f"LV{mod[1]}_constant")


def test_Poisson(loss, pinn, loss_values, t_domain, mod=["", ""]):
    # Result of training
    print_loss(loss, pinn)
    plot_loss(loss_values, title=f"Poisson problem{mod[0]}", save=f"Poisson{mod[1]}_loss")
    t = torch.linspace(t_domain[0], t_domain[1], 1001).reshape(-1, 1).to(pinn.device())
    t.requires_grad = True
    plot_1D(pinn, t, title=f"Poisson problem{mod[0]}", xlabel="x", ylabel="Phi", save=f"Poisson{mod[1]}_pinn_result")

    # FEM
    Y_FEM, times = FEM_Poisson(3001)

    # PINN
    (Y_pinn, ) = get_values_from_pinn(pinn, times)

    plot_compare([Y_FEM, Y_pinn], times, ["FEM", "PINN"], title="Poisson equation for gravity", xlabel="x", ylabel="Phi", save=f"Poisson{mod[1]}")


def test_Heat(loss, pinn, loss_values, x_domain, y_domain, t_domain, mod=["", ""]):
    # Result of training
    print_loss(loss, pinn)
    plot_loss(loss_values, title=f"Heat problem{mod[0]}", save=f"Heat{mod[1]}_loss")

    # IGA
    data = get_data("./other_methods/IGA")
    # plot_3D_IGA(data, name="3D_IGA")
    times = []
    c_IGA = []
    for (t, x, y, z) in data:
        N = int(round(len(x) ** 0.5))
        times.append(t)
        c_IGA.append(get_Heat_level(torch.tensor(z)))
    c_IGA = torch.tensor(c_IGA)
    times = torch.tensor(times)

    x = torch.linspace(x_domain[0], x_domain[1], N).reshape(-1, 1).to(pinn.device())
    x.requires_grad = True
    y = torch.linspace(y_domain[0], y_domain[1], N).reshape(-1, 1).to(pinn.device())
    y.requires_grad = True
    t = torch.linspace(t_domain[0], t_domain[1], len(times)).reshape(-1, 1).to(pinn.device())
    t.requires_grad = True

    plot_3D(pinn, x, y, t, name="3D_PINN")

    grid_x, grid_y = torch.meshgrid(x.reshape(-1), y.reshape(-1), indexing='ij')
    grid_x = grid_x.reshape(-1, 1)
    grid_y = grid_y.reshape(-1, 1)

    c_pinn = []
    for t0 in times:
        c_pinn.append(get_Heat_level(pinn(grid_x, grid_y, torch.full_like(grid_x, t0))))
    c_pinn = torch.tensor(c_pinn)

    plot_difference([c_IGA, c_pinn], times, get_Heat_start_level(grid_x, grid_y).detach().cpu().numpy(), ["IGA", "PINN"], title="Constant in Heat", ylabel="Difference", save=f"Heat{mod[1]}_constant")
