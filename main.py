from train_models import train_SIR, train_Kepler, train_VL
from test import test_SIR, test_Kepler, test_VL

if __name__ == "__main__":
    t_domain_SIR = [0, 10]

    t_domain_Kepler = [0, 20]

    t_domain_VL = [0, 10]

    loss_SIR, best_pinn_SIR, loss_values_SIR = train_SIR(t_domain_SIR, epochs=10)
    loss_Kepler, best_pinn_Kepler, loss_values_Kepler = train_Kepler(t_domain_Kepler, help=True, epochs=10)
    loss_VL, best_pinn_VL, loss_values_VL = train_VL(t_domain_VL, help=True, epochs=10)

    test_SIR(loss_SIR, best_pinn_SIR, loss_values_SIR, t_domain_SIR)
    test_Kepler(loss_Kepler, best_pinn_Kepler, loss_values_Kepler, t_domain_Kepler)
    test_VL(loss_VL, best_pinn_VL, loss_values_VL, t_domain_VL)
