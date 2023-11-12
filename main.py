from train import train_SIR, train_Gravity, train_Tsunami
from test import test_SIR, test_Gravity, test_Tsunami


if __name__ == "__main__":
    t_domain_SIR = [0, 10]

    t_domain_Gravity = [0, 20]

    x_domain_Tsunami = [0, 1]
    y_domain_Tsunami = [0, 1]
    t_domain_Tsunami = [0, 1]

    loss_SIR, best_pinn_SIR, loss_values_SIR = train_SIR(t_domain_SIR, epochs=10)
    loss_Gravity, best_pinn_Gravity, loss_values_Gravity = train_Gravity(t_domain_Gravity, epochs=10)
    # loss_Tsunami, best_pinn_Tsunami, loss_values_Tsunami = train_Tsunami(x_domain_Tsunami, y_domain_Tsunami, t_domain_Tsunami)

    test_SIR(loss_SIR, best_pinn_SIR, loss_values_SIR, t_domain_SIR)
    test_Gravity(loss_Gravity, best_pinn_Gravity, loss_values_Gravity, t_domain_Gravity)
    # test_Tsunami(loss_Tsunami, best_pinn_Tsunami, loss_values_Tsunami, x_domain_Tsunami, y_domain_Tsunami, t_domain_Tsunami)