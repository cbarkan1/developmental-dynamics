import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def deterministic_equations(N, t):
    nS, nT, nD = N[:]
    dSdt = beta_SSS(nS)*nS - F_Slost(nD)*nS + beta_Sstar*nT - mu*nS
    dTdt = F_Slost(nD)*nS - beta_Sstar*nT - mu*nT
    dDdt = beta_Sstar*nT - mu*nD
    return np.array([dSdt, dTdt, dDdt])

def beta_SSS(nS):
    return 1/(1 + 0.15*np.exp(nS*0.5))

def F_Slost(nD):
    """ F_{\mathcal{S},\text{lost}}, the rate of stem cell triggering."""
    return np.maximum(np.zeros_like(nD), -7/5 * (nD-26))


if __name__=="__main__":
    mu = 0.02
    beta_Sstar = 0.6

    N0 = np.array([4, 0, 0])
    ts = np.linspace(0, 40, 500)
    N_of_t = odeint(deterministic_equations, N0, ts)

    nS_of_t = N_of_t[:, 0]
    nT_of_t = N_of_t[:, 1]
    nD_of_t = N_of_t[:, 2]
    fig, a = plt.subplots(1,1,figsize=(4,4))
    a.plot(ts, nS_of_t, label='$N_S$')
    a.plot(ts, nT_of_t, label='$N_{S^T}$')
    a.plot(ts, nD_of_t, label='$N_D$')
    #plt.legend()
    a.set_ylim(0, 43)
    #a.set_xlabel('t')
    #a.set_ylabel('Population')
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.savefig("Fig3B.svg",transparent=True)
    plt.show()
