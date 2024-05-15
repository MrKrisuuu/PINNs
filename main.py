from training.train_models import train_SIR, train_Kepler, train_LV
from test import test_SIR, test_Kepler, test_LV
from constants.constants_Kepler import get_Kepler_start_energy, get_Kepler_start_moment

if __name__ == "__main__":
    t_domain_SIR = [0, 10]

    t_domain_Kepler = [0, 20]

    t_domain_LV = [0, 10]

    loss_SIR, best_pinn_SIR, loss_values_SIR = train_SIR(t_domain_SIR)
    test_SIR(loss_SIR, best_pinn_SIR, loss_values_SIR, t_domain_SIR)

    # loss_Kepler, best_pinn_Kepler, loss_values_Kepler = train_Kepler(t_domain_Kepler, pretrain_epochs=10000)
    # test_Kepler(loss_Kepler, best_pinn_Kepler, loss_values_Kepler, t_domain_Kepler)
    #
    # loss_LV, best_pinn_LV, loss_values_LV = train_LV(t_domain_LV)
    # test_LV(loss_LV, best_pinn_LV, loss_values_LV, t_domain_LV)
    #
    # loss_Kepler, best_pinn_Kepler, loss_values_Kepler = train_Kepler(t_domain_Kepler, invariant=True)
    # test_Kepler(loss_Kepler, best_pinn_Kepler, loss_values_Kepler, t_domain_Kepler, mod=[" with Invariant loss", "_invariant"])
    #
    # loss_LV, best_pinn_LV, loss_values_LV = train_LV(t_domain_LV, invariant=True)
    # test_LV(loss_LV, best_pinn_LV, loss_values_LV, t_domain_LV, mod=[" with Invariant loss", "_invariant"])

    # loss_Kepler, best_pinn_Kepler, loss_values_Kepler = train_Kepler(t_domain_Kepler, invariant=True, pretrain_epochs=10000)
    # test_Kepler(loss_Kepler, best_pinn_Kepler, loss_values_Kepler, t_domain_Kepler, mod=[" with pre-training", "_pre"])

