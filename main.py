from training.train_models import train_SIR, train_Kepler, train_LV, train_Poisson, train_Heat
from test import test_SIR, test_Kepler, test_LV, test_Poisson, test_Heat
import torch

if __name__ == "__main__":
    t_domain_SIR = [0, 10]

    t_domain_Kepler = [0, 20]

    t_domain_LV = [0, 10]

    t_domain_Poisson = [0, 3]

    x_domain_Heat = [0, 1]
    y_domain_Heat = [0, 1]
    t_domain_Heat = [0, 0.1]


    loss_SIR, best_pinn_SIR, loss_values_SIR = train_SIR(t_domain_SIR)
    test_SIR(loss_SIR, best_pinn_SIR, loss_values_SIR, t_domain_SIR)

    # loss_Kepler, best_pinn_Kepler, loss_values_Kepler = train_Kepler(t_domain_Kepler, pretrain_epochs=10000)
    # test_Kepler(loss_Kepler, best_pinn_Kepler, loss_values_Kepler, t_domain_Kepler)

    # loss_LV, best_pinn_LV, loss_values_LV = train_LV(t_domain_LV)
    # test_LV(loss_LV, best_pinn_LV, loss_values_LV, t_domain_LV)

    # loss_Kepler, best_pinn_Kepler, loss_values_Kepler = train_Kepler(t_domain_Kepler, invariant=True)
    # test_Kepler(loss_Kepler, best_pinn_Kepler, loss_values_Kepler, t_domain_Kepler, mod=[" with Invariant loss", "_invariant"])

    # loss_LV, best_pinn_LV, loss_values_LV = train_LV(t_domain_LV, invariant=True)
    # test_LV(loss_LV, best_pinn_LV, loss_values_LV, t_domain_LV, mod=[" with Invariant loss", "_invariant"])

    # loss_Kepler, best_pinn_Kepler, loss_values_Kepler = train_Kepler(t_domain_Kepler, invariant=True, pretrain_epochs=10000)
    # test_Kepler(loss_Kepler, best_pinn_Kepler, loss_values_Kepler, t_domain_Kepler, mod=[" with pre-training", "_pre"])

    # loss_Kepler, best_pinn_Kepler, loss_values_Kepler = train_Kepler(t_domain_Kepler, invariant=True, epochs=20000, pretrain_epochs=10000, LBFGS_epochs=10000)
    # test_Kepler(loss_Kepler, best_pinn_Kepler, loss_values_Kepler, t_domain_Kepler, mod=[" with L-BFGS", "_LBFGS"])

    # loss_Poisson, best_pinn_Poisson, loss_values_Poisson = train_Poisson(t_domain_Poisson, LBFGS_epochs=2000)
    # test_Poisson(loss_Poisson, best_pinn_Poisson, loss_values_Poisson, t_domain_Poisson)

    # loss_Heat, best_pinn_Heat, loss_values_Heat = train_Heat(x_domain_Heat, y_domain_Heat, t_domain_Heat, epochs=50000, LBFGS_epochs=20000, invariant=True)
    # test_Heat(loss_Heat, best_pinn_Heat, loss_values_Heat, x_domain_Heat, y_domain_Heat, t_domain_Heat)

