import torch

from PINN import PINN
from Loss import Loss_SIR, Loss_Gravity, Loss_Tsunami
from train import train_model
from plotting import print_loss, plot_loss, plot_1D, plot_1D_in_2D, plot_2D, plot_3D
from other_plotting import plot_SIR_number, plot_SIR_change, plot_Gravity_energy, plot_Gravity_momentum, plot_Tsunami_level

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)





# # SIR problem
# pinn_SIR = PINN(1, 3).to(device)
#
# t_domain_SIR = [0, 20]
#
# loss_SIR = Loss_SIR(
#     t_domain_SIR,
#     n_points=200
# )
#
# best_pinn_SIR, loss_values_SIR = train_model(
#     pinn_SIR, loss_fn=loss_SIR, max_epochs=10000)
# best_pinn_SIR = best_pinn_SIR.cpu()
#
# torch.save(best_pinn_SIR, "./results/SIR.pth")





# # Gravity problem
# pinn_Gravity = PINN(1, 2).to(device)

t_domain_Gravity = [0, 50]

loss_Gravity = Loss_Gravity(
    t_domain_Gravity,
    n_points=2000
)
#
# best_pinn_Gravity, loss_values_Gravity = train_model(
#     pinn_Gravity, loss_fn=loss_Gravity, max_epochs=50000)
# best_pinn_Gravity = best_pinn_Gravity.cpu()
#
# torch.save(best_pinn_Tsunami, "./results/Gravity.pth")





# # Tsunami problem
# pinn_Tsunami = PINN(3, 1).to(device)
#
x_domain_Tsunami = [0, 1]
y_domain_Tsunami = [0, 1]
t_domain_Tsunami = [0, 1]

loss_Tsunami = Loss_Tsunami(
    x_domain_Tsunami,
    y_domain_Tsunami,
    t_domain_Tsunami,
    n_points=15
)
#
# best_pinn_Tsunami, loss_values_Tsunami = train_model(
#     pinn_Tsunami, loss_fn=loss_Tsunami, max_epochs=50000)
# best_pinn_Tsunami = best_pinn_Tsunami.cpu()
#
# torch.save(best_pinn_Gravity, "./results/Tsunami.pth")





# print_loss(loss_SIR, best_pinn_SIR)
# plot_loss(loss_values_SIR, name="loss_SIR")
#
# t = torch.linspace(t_domain_SIR[0], t_domain_SIR[1], 101).reshape(-1, 1)
# t.requires_grad = True
#
# plot_1D(best_pinn_SIR, t, name="SIR")
# plot_SIR_number(best_pinn_SIR, t)
# plot_SIR_change(best_pinn_SIR, t)


# print_loss(loss_Gravity, best_pinn_Gravity)
# #plot_loss(loss_values_Gravity, name="loss_Gravity")
#
# t = torch.linspace(t_domain_Gravity[0], t_domain_Gravity[1], 101).reshape(-1, 1)
# t.requires_grad = True
#
# plot_1D_in_2D(best_pinn_Gravity, t, name="Gravity")
# plot_1D(best_pinn_Gravity, t, name="Gravity (sins)")
# plot_Gravity_energy(best_pinn_Gravity, t)
# plot_Gravity_momentum(best_pinn_Gravity, t)


best_pinn_Tsunami = torch.load("./results/Tsunami.pth")

print_loss(loss_Tsunami, best_pinn_Tsunami)
#plot_loss(loss_values_Tsunami, name="loss_Tsunami")


x = torch.linspace(x_domain_Tsunami[0], x_domain_Tsunami[1], 101).reshape(-1, 1)
x.requires_grad = True
y = torch.linspace(y_domain_Tsunami[0], y_domain_Tsunami[1], 101).reshape(-1, 1)
y.requires_grad = True
t = torch.linspace(t_domain_Tsunami[0], t_domain_Tsunami[1], 101).reshape(-1, 1)
t.requires_grad = True

plot_3D(best_pinn_Tsunami, x, y, t, name="Tsunami")
plot_Tsunami_level(best_pinn_Tsunami, x, y, t)


