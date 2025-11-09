import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from Example_1_Model import Example_1_Model, parameters
from deterministic_model import F_Slost

def estimate_mu_clopper_pearson(y, dt, alpha=0.05):
    """
    y : 1D array of counts y_t
    dt: time step length
    alpha: 1 - confidence level (0.05 for 95% CI)
    """

    # Total trials and total survivals
    M = np.sum(y[0:-1])
    S = np.sum(y[1:])

    # MLE for p and mu
    p_hat = S / M

    mu_hat = -np.log(p_hat) / dt

    # Clopper-Pearson CI for p
    a = alpha / 2.0

    if 0 < S < M:
        # Interior case
        p_L = beta.ppf(a,     S,   M - S + 1)
        p_U = beta.ppf(1 - a, S+1, M - S)
    elif S == 0:
        # No successes
        p_L = 0.0
        p_U = beta.ppf(1 - a, 1, M)
    elif S == M:
        # All successes
        p_L = beta.ppf(a, M, 1)
        p_U = 1.0
    else:
        raise RuntimeError("Logic error in S, M handling.")

    # Transform to mu: note order flips since mu decreases with p
    mu_L = -np.log(p_U) / dt if p_U > 0 else np.inf # Lower bound of CI
    mu_U = -np.log(p_L) / dt if p_L > 0 else np.inf # Upper bound of CI

    return mu_hat, mu_L, mu_U


def estimate_transition_rate(q,nS_0, nSstar_0, nD_0):
    """
    Estimate the transition rate for cells transitioning out of type q
    """
    model = Example_1_Model(parameters, max_cell_num=nS_0+nSstar_0+nD_0)  
    for i in range(nS_0):
        model.create_cell(1, 20)
    for i in range(nSstar_0):
        model.create_cell(10, 10)
    for i in range(nD_0):
        model.create_cell(30, 1)

    living_indices = np.where(model.Pop[:, 2] == 1)[0]
    Ns = model.find_cell_type_numbers(living_indices)

    ns = np.zeros((3,NUM_STEPS))
    ns[:,0] = [nS_0, nSstar_0, nD_0]
    for step in range(1,NUM_STEPS):
        c_1, c_2 = model.find_signaling_molecule_concentrations(living_indices)
        model.update_gene_state(dt, living_indices, c_1)
        Ns = model.find_cell_type_numbers(living_indices)
        ns[:,step] = Ns[:]

    cell_type_dict = {"S":0,"Sprime":1,"D":2}
    F_out, F_out_L, F_out_U = estimate_mu_clopper_pearson(ns[cell_type_dict[q],:],dt)
    return F_out, F_out_L, F_out_U


if __name__=="__main__":
    dt = 0.01 # Timestep in days
    NUM_STEPS = 1000
    
    cell_type = "S"

    nS_0 = 100
    nSstar_0 = 0
    nDs = np.arange(0,40)
    F_outs = np.zeros(len(nDs))
    F_outs_L = np.zeros(len(nDs))
    F_outs_U = np.zeros(len(nDs))
    for i,nD in enumerate(nDs):
        F_outs[i], F_outs_L[i], F_outs_U[i] = estimate_transition_rate(cell_type,nS_0, nSstar_0, nD)
        print(nD, F_outs[i], F_outs_L[i], F_outs_U[i])
    
    max_other_rate = 1/(1+0.15) # 1/day
    
    plt.figure(figsize=(3,2))
    plt.fill_between(nDs,F_outs_U,F_outs_L,alpha=0.3)
    plt.plot(nDs, F_outs)
    xlim = plt.gca().get_xlim()
    plt.plot(xlim,[max_other_rate,max_other_rate],"k:")
    plt.plot(np.array([1,26,50]), F_Slost(np.array([1,26,50])), '--',color="grey")
    plt.xlim([0,xlim[1]])
    plt.ylim(0,7)
    plt.savefig("determ_appendix.pdf",transparent=True)
    plt.show()

