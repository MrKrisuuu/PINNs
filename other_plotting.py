from PINN import f, dfdt, dfdx, dfdy
import matplotlib.pyplot as plt
import torch


def plot_SIR_change(pinn, t):
    S = dfdt(pinn, t, output_value=0)
    I = dfdt(pinn, t, output_value=1)
    R = dfdt(pinn, t, output_value=2)
    val = S + I + R
    plt.plot(t.detach().cpu().numpy(), val.detach().cpu().numpy())
    plt.savefig("./results/plot_SIR_change.png")
    plt.show()


def plot_SIR_number(pinn, t):
    S = f(pinn, t, output_value=0)
    I = f(pinn, t, output_value=1)
    R = f(pinn, t, output_value=2)
    val = S + I + R
    plt.plot(t.detach().cpu().numpy(), val.detach().cpu().numpy())
    plt.savefig("./results/plot_SIR_number.png")
    plt.show()


def plot_Gravity_energy(pinn, t):
    r = (f(pinn, t, output_value=0) ** 2 + f(pinn, t, output_value=1) ** 2) ** (1 / 2)
    val = (dfdt(pinn, t, output_value=0) ** 2 + dfdt(pinn, t, output_value=1) **2) / 2 - 1 / r
    plt.plot(t.detach().cpu().numpy(), val.detach().cpu().numpy())
    plt.savefig("./results/plot_Gravity_energy.png")
    plt.show()


def plot_Gravity_momentum(pinn, t):
    val = f(pinn, t, output_value=0) * dfdt(pinn, t, output_value=1) - \
          f(pinn, t, output_value=1) * dfdt(pinn, t, output_value=0)
    plt.plot(t.detach().cpu().numpy(), val.detach().cpu().numpy())
    plt.savefig("./results/plot_Gravity_momentum.png")
    plt.show()


def plot_Tsunami_level(pinn, x, y, t):
    vals = []
    for t_raw in t:
        t0 = torch.full_like(x, t_raw.item())
        val = f(pinn, x, y, t0).mean()
        vals.append(val.detach().cpu().numpy())
    plt.plot(t.detach().cpu().numpy(), vals)
    plt.savefig("./results/plot_Tsunami_level.png")
    plt.show()