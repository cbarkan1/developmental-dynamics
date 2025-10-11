import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def Waddington(N, t):
    nS, nT, nD = N[:]
    dSdt = betaS(nS)*nS - F(nD)*nS + betaT*nT - mu*nS
    dTdt = F(nD)*nS - betaT*nT - mu*nT
    dDdt = betaT*nT - mu*nD
    return np.array([dSdt, dTdt, dDdt])


def betaS(nS):
    return 7./(7. + np.exp(nS*0.5))


def F(nD):
    """ Rate of stem cell triggering."""
    if nD > 25:
        return 0.
    else:
        return 0.1*(25. - nD)


def F2(nD):
    """ Rate of stem cell triggering."""
    return 1./(1. + np.exp(-(25-nD)))


mu = 0.03
betaT = 0.5

N0 = np.array([4, 0, 0])
ts = np.linspace(0, 40, 500)
N_of_t = odeint(Waddington, N0, ts)

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
plt.show()
