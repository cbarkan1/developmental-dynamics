"""
This new file will be paired with model_utils.py

Simulates a minimal developmental model inspired by hematopoietic stem
cell differentiation.

"""

import numpy as np
import matplotlib.pyplot as plt
from model_utils import Model

max_cell_num = 100  # Max number of cells that can be alive at a time

dt = 0.1
num_steps = 2000
ts = np.arange(0, dt*num_steps, dt)

print('time simulated = ', num_steps*dt)

# Initial model:
S_th = 1.5  # xB threshold
D_th = 0.5  # xB threshold

r_R1 = 0.5  # * 0.4
alpha1 = 20
r_01 = 1
r_Rt1 = 2  # * .4 #1
r_t1 = 3  # Expression rate with only t.f. bound.
k1 = 1

r_R2 = 0
r_02 = 1
r_Rt2 = 1
r_t2 = 2  # Expression rate with only t.f. bound.
k2 = 1

n = 4
a = .5**n

"""
#These parameters give Huang's 3 state vector field
r_R1 = 0
r_01 = 1.
r_Rt1 = 1.
r_t1 = r_01 + r_Rt1 # Expression rate with only t.f. bound.
k1 = 1
nD = 10

r_R2 = 0
r_02 = 1.
r_Rt2 = 1.
r_t2 = r_02 + r_Rt2 # Expression rate with only t.f. bound.
k2 = 1

n = 4
a = .5**n
"""

model = Model(S_th, D_th, r_R1, alpha1, r_01, r_Rt1, 
              r_t1, k1, r_R2, r_02, r_Rt2, r_t2, k2, n, a)

# Initial population:
Pop = np.zeros((max_cell_num, 3))  # properties: (xA,xB,alive)
Pop[0, :] = [.1, 2.5, 1]
Pop[1, :] = [0.2, 2., 1]
Pop[2, :] = [0.15, 1.5, 1]
Pop[3, :] = [.05, .5, 1]

living_indices = np.where(Pop[:, 2] == 1)[0]
nonliving_indices = np.where(Pop[:, 2] == 0)[0]

model.plot_Pop(Pop, living_indices)

N_Ss = np.zeros(num_steps)
N_Sts = np.zeros(num_steps)
N_Ds = np.zeros(num_steps)
for step in range(num_steps):
    N_Ss[step], N_Sts[step], N_Ds[step] = model.find_Ns(
        Pop, living_indices=living_indices)

    # Lambda dynamics
    # Put this in the Model class
    lambdaS = np.sum(Pop[living_indices, 1] > S_th)
    lambdaD = np.sum(Pop[living_indices, 1] < D_th)

    # Gene Dynamics
    V0, V1 = model.V(Pop[living_indices, 0], Pop[living_indices, 1], lambdaD)
    Pop[living_indices, 0] += dt*V0 + 0.01*Pop[living_indices, 0] \
        * np.sqrt(dt)*np.random.normal(size=len(living_indices))
    Pop[living_indices, 1] += dt*V1 + 0.01*Pop[living_indices, 1] \
        * np.sqrt(dt)*np.random.normal(size=len(living_indices))

    # Cell division
    betas = model.beta(Pop[living_indices, 1], lambdaS, lambdaD)
    division_indices = living_indices[
        np.random.uniform(size=len(living_indices)) < betas*dt]
    for i, index in enumerate(division_indices):
        if Pop[index, 1] > S_th:  # Symmetric
            Pop[nonliving_indices[i], :] = Pop[index, :].copy()
        elif Pop[index, 1] >= D_th:  # Assymetric
            d_xA = 0.95*Pop[index, 0]
            d_xB = 0.95*Pop[index, 1]
            Pop[nonliving_indices[i], :] = [
                Pop[index, 0]+d_xA, Pop[index, 1]-d_xB, 1]
            Pop[index, :] = [Pop[index, 0]-d_xA, Pop[index, 1]+d_xB, 1]

    living_indices = np.where(Pop[:, 2] == 1)[0]

    # Cell death
    death_indices = living_indices[
        np.random.uniform(size=len(living_indices)) < 0.03*dt]
    Pop[death_indices, 2] = 0.
    living_indices = np.where(Pop[:, 2] == 1)[0]
    nonliving_indices = np.where(Pop[:, 2] == 0)[0]

model.plot_Pop(Pop, living_indices)
plt.figure()
plt.plot(ts, N_Ss)
plt.plot(ts, N_Sts)
plt.plot(ts, N_Ds)
plt.plot([-1, ts[-1]+1], [26, 26], 'k', linewidth=1)
plt.plot([-1, ts[-1]+1], [33, 33], 'k', linewidth=1)
plt.xlim(-1, ts[-1]+1)
plt.ylim(0, 43)
plt.legend(['$N_S$', '$N_{S^T}$', '$N_D$'], loc='upper left')
plt.gca().spines[['right', 'top']].set_visible(False)
plt.show()
