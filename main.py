from training.train_models import train_SIR, train_Kepler, train_LV
from test import test_SIR, test_Kepler, test_LV

if __name__ == "__main__":
    t_domain_SIR = [0, 10]

    t_domain_Kepler = [0, 20]

    t_domain_LV = [0, 10]

    # loss_SIR, best_pinn_SIR, loss_values_SIR = train_SIR(t_domain_SIR, help=True, epochs=1000)
    # loss_Kepler, best_pinn_Kepler, loss_values_Kepler = train_Kepler(t_domain_Kepler, help=True, epochs=1000)
    loss_LV, best_pinn_LV, loss_values_LV = train_LV(t_domain_LV, help=True, epochs=1000)

    # test_SIR(loss_SIR, best_pinn_SIR, loss_values_SIR, t_domain_SIR)
    # test_Kepler(loss_Kepler, best_pinn_Kepler, loss_values_Kepler, t_domain_Kepler)
    test_LV(loss_LV, best_pinn_LV, loss_values_LV, t_domain_LV)
