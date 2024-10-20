import torch

from other_methods.other_methods_Poisson import FEM_Poisson
from other_methods.other_methods_IGA import get_data
from other_methods.methods import euler, implicit_euler, rk4, semi_euler, verlet
from other_methods.equations import SIR, Kepler, LV

from utils import get_times, get_derivatives, get_values_from_pinn, get_derivatives_from_pinn
from PINN import dfdt
from plotting import print_loss, plot_loss, plot_1D, plot_1D_in_2D, plot_compare, plot_difference
from other_plotting import plot_3D, plot_3D_IGA

from constants.constants_SIR import get_SIR_start_sum, get_SIR_sum, get_SIR_start_constant, get_SIR_constant
from constants.constants_Kepler import get_Kepler_start_energy, get_Kepler_energy, get_Kepler_start_moment, get_Kepler_moment
from constants.constants_LV import get_LV_start_c, get_LV_c
from constants.constants_Heat import get_Heat_start_level, get_Heat_level
from constants.initial_conditions import get_initial_conditions


def test_SIR(loss, pinn, loss_values, t_domain, h=0.001):
    # Result of training
    print_loss(loss, pinn)
    plot_loss(loss_values, save="SIR_loss")
    t = torch.linspace(t_domain[0], t_domain[1], 1001).reshape(-1, 1).to(pinn.device())
    t.requires_grad = True
    plot_1D(pinn, t, save="SIR_pinn_result", labels=["S", "I", "R"], ylabel="Population")

    times = get_times(t_domain[1], h)

    # Euler
    r_euler = euler(SIR, times, get_initial_conditions("SIR")[0], params=get_initial_conditions("SIR")[1])
    n_euler = get_SIR_sum(r_euler)
    c_euler = get_SIR_constant(r_euler)

    # Implicit Euler
    r_implicit = implicit_euler(SIR, times, get_initial_conditions("SIR")[0], params=get_initial_conditions("SIR")[1])
    n_implicit = get_SIR_sum(r_implicit)
    c_implicit = get_SIR_constant(r_implicit)

    # RK4
    r_rk4 = rk4(SIR, times, get_initial_conditions("SIR")[0], params=get_initial_conditions("SIR")[1])
    n_rk4 = get_SIR_sum(r_rk4)
    c_rk4 = get_SIR_constant(r_rk4)

    # PINN
    r_pinn = get_values_from_pinn(pinn, times)
    n_pinn = get_SIR_sum(r_pinn)
    c_pinn = get_SIR_constant(r_pinn)

    # Compare methods
    plot_compare([r_euler[:, 0], r_implicit[:, 0], r_rk4[:, 0], r_pinn[:, 0]], times, ["Euler", "Implicit", "RK4", "PINN"], ylabel="Susceptible individuals", save="SIR_S")
    plot_compare([r_euler[:, 1], r_implicit[:, 1], r_rk4[:, 1], r_pinn[:, 1]], times, ["Euler", "Implicit", "RK4", "PINN"], ylabel="Infectious individuals", save="SIR_I")
    plot_compare([r_euler[:, 2], r_implicit[:, 2], r_rk4[:, 2], r_pinn[:, 2]], times, ["Euler", "Implicit", "RK4", "PINN"], ylabel="Removed individuals", save="SIR_R")
    plot_difference([n_euler, n_implicit, n_rk4, n_pinn], times, torch.full_like(times, get_SIR_start_sum()), ["Euler", "Implicit", "RK4", "PINN"], save="SIR_total")
    plot_difference([c_euler, c_implicit, c_rk4, c_pinn], times, torch.full_like(times, get_SIR_start_constant()), ["Euler", "Implicit", "RK4", "PINN"], save="SIR_constant")


def test_Kepler(loss, pinn, loss_values, t_domain, h=0.001, mod=""):
    # Results of training
    print_loss(loss, pinn)
    plot_loss(loss_values, save=f"Kepler{mod}_loss")
    t = torch.linspace(t_domain[0], t_domain[1], 1001).reshape(-1, 1).to(pinn.device())
    t.requires_grad = True
    plot_1D_in_2D(pinn, t, save=f"Kepler{mod}_pinn_orbit")
    plot_1D(pinn, t, labels=["X", "Y"], ylabel="Value", save=f"Kepler{mod}_pinn_result")

    times = get_times(t_domain[1], h)

    # Euler
    r_euler = euler(Kepler, times, get_initial_conditions("Kepler")[0], params=get_initial_conditions("Kepler")[1])
    energy_euler = get_Kepler_energy(r_euler)
    momentum_euler = get_Kepler_moment(r_euler)

    # Semi-implicit Euler
    r_semi = semi_euler(Kepler, times, get_initial_conditions("Kepler")[0], params=get_initial_conditions("Kepler")[1])
    energy_semi = get_Kepler_energy(r_semi)
    momentum_semi = get_Kepler_moment(r_semi)

    # Implicit Euler
    r_implicit = implicit_euler(Kepler, times, get_initial_conditions("Kepler")[0], params=get_initial_conditions("Kepler")[1])
    energy_implicit = get_Kepler_energy(r_implicit)
    momentum_implicit = get_Kepler_moment(r_implicit)

    # RK
    r_rk4 = rk4(Kepler, times, get_initial_conditions("Kepler")[0], params=get_initial_conditions("Kepler")[1])
    energy_RK = get_Kepler_energy(r_rk4)
    momentum_RK = get_Kepler_moment(r_rk4)

    # Verlet
    r_verlet = verlet(Kepler, times, get_initial_conditions("Kepler")[0], params=get_initial_conditions("Kepler")[1])
    energy_Verlet = get_Kepler_energy(r_verlet)
    momentum_Verlet = get_Kepler_moment(r_verlet)

    # PINN
    r_pinn = get_values_from_pinn(pinn, times)
    dX_pinn = get_derivatives_from_pinn(pinn, times, dfdt, output_value=0).reshape(-1, 1)
    dY_pinn = get_derivatives_from_pinn(pinn, times, dfdt, output_value=1).reshape(-1, 1)
    r_pinn = torch.concat((r_pinn, dX_pinn, dY_pinn), dim=1)
    energy_pinn = get_Kepler_energy(r_pinn)
    momentum_pinn = get_Kepler_moment(r_pinn)

    # Compare methods
    plot_compare([r_euler[:, 0], r_semi[:, 0], r_implicit[:, 0], r_rk4[:, 0], r_verlet[:, 0], r_pinn[:, 0]], times, ["Euler", "Semi", "Implicit", "RK4", "Verlet", "PINN"], ylabel="X", save=f"Kepler{mod}_X")
    plot_compare([r_euler[:, 1], r_semi[:, 1], r_implicit[:, 1], r_rk4[:, 1], r_verlet[:, 1], r_pinn[:, 1]], times, ["Euler", "Semi", "Implicit", "RK4", "Verlet", "PINN"], ylabel="Y", save=f"Kepler{mod}_Y")
    plot_difference([energy_euler, energy_semi, energy_implicit, energy_RK, energy_Verlet, energy_pinn], times, get_Kepler_start_energy(), ["Euler", "Semi", "Implicit", "RK4", "Verlet", "PINN"], save=f"Kepler{mod}_energy")
    plot_difference([momentum_euler, momentum_semi, momentum_implicit, momentum_RK, momentum_Verlet, momentum_pinn], times, get_Kepler_start_moment(), ["Euler", "Semi", "Implicit", "RK4", "Verlet", "PINN"], save=f"Kepler{mod}_momentum")


def test_LV(loss, pinn, loss_values, t_domain, h=0.001, mod=""):
    # Result of training
    print_loss(loss, pinn)
    plot_loss(loss_values, save=f"LV{mod}_loss")
    t = torch.linspace(t_domain[0], t_domain[1], 1001).reshape(-1, 1).to(pinn.device())
    t.requires_grad = True
    plot_1D(pinn, t, labels=["Preys", "Predators"], ylabel="Population", save=f"LV{mod}_pinn_result")

    times = get_times(t_domain[1], h)

    # Euler
    r_euler = euler(LV, times, get_initial_conditions("LV")[0], params=get_initial_conditions("LV")[1])
    c_euler = get_LV_c(r_euler)

    # Implicit Euler
    r_implicit = implicit_euler(LV, times, get_initial_conditions("LV")[0], params=get_initial_conditions("LV")[1])
    c_implicit = get_LV_c(r_implicit)

    # RK
    r_rk4 = rk4(LV, times, get_initial_conditions("LV")[0], params=get_initial_conditions("LV")[1])
    c_rk4 = get_LV_c(r_rk4)

    # PINN
    r_pinn = get_values_from_pinn(pinn, times)
    c_pinn = get_LV_c(r_pinn)

    # Compare methods
    plot_compare([r_euler[:, 0], r_implicit[:, 0], r_rk4[:, 0], r_pinn[:, 0]], times, ["Euler", "Implicit", "RK4", "PINN"], ylabel="Prey individuals", save=f"LV{mod}_prey")
    plot_compare([r_euler[:, 1], r_implicit[:, 1], r_rk4[:, 1], r_pinn[:, 1]], times, ["Euler", "Implicit", "RK4", "PINN"], ylabel="Predators individuals", save=f"LV{mod}_predator")
    plot_difference([c_euler, c_implicit, c_rk4, c_pinn], times, get_LV_start_c(), ["Euler", "Implicit", "RK4", "PINN"], save=f"LV{mod}_constant")


def test_Poisson(loss, pinn, loss_values, t_domain, mod=""):
    # Result of training
    print_loss(loss, pinn)
    plot_loss(loss_values, save=f"Poisson{mod}_loss")
    t = torch.linspace(t_domain[0], t_domain[1], 1001).reshape(-1, 1).to(pinn.device())
    t.requires_grad = True
    plot_1D(pinn, t, xlabel="x", ylabel="Phi", save=f"Poisson{mod}_pinn_result")

    # FEM
    r_FEM, times = FEM_Poisson(3001)

    # PINN
    r_pinn = get_values_from_pinn(pinn, times)

    plot_compare([r_FEM, r_pinn], times, ["FEM", "PINN"], xlabel="x", ylabel="Phi", save=f"Poisson{mod}")


def test_Heat(loss, pinn, loss_values, x_domain, y_domain, t_domain, mod=""):
    # Result of training
    print_loss(loss, pinn)
    plot_loss(loss_values, save=f"Heat{mod}_loss")

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

    plot_difference([c_IGA, c_pinn], times, get_Heat_start_level(grid_x, grid_y).detach().cpu().numpy(), ["IGA", "PINN"], save=f"Heat{mod}_constant")
