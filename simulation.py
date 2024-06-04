"""

Simulates a minimal developmental model inspired by hematopoietic stem
cell differentiation.

"""

import numpy as np
import matplotlib.pyplot as plt
from model_utils import Model
from parameters import params

dt = 0.1
num_steps = 2000
ts = np.arange(0, dt*num_steps, dt)

print('time simulated = ', num_steps*dt)

model = Model(*params)
model.create_Pop(max_cell_num=100)
model.create_cell(0.1, 2.5)
model.create_cell(0.2, 2.)
model.create_cell(0.15, 1.5)
model.create_cell(0.05, .5)

model.plot_Pop()


N_Ss = np.zeros(num_steps)
N_Sts = np.zeros(num_steps)
N_Ds = np.zeros(num_steps)
for step in range(num_steps):
    living_indices = np.where(model.Pop[:, 2] == 1)[0]
    nonliving_indices = np.where(model.Pop[:, 2] == 0)[0]

    N_Ss[step], N_Sts[step], N_Ds[step] = model.find_Ns(
        living_indices=living_indices)

    # Lambda dynamics
    lambdaS, lambdaD = model.find_lambdas(living_indices)

    # Gene Dynamics
    V0, V1 = model.V(model.Pop[living_indices, 0],
                     model.Pop[living_indices, 1], lambdaD)
    model.Pop[living_indices, 0] += dt*V0 + 0.01*model.Pop[living_indices, 0] \
        * np.sqrt(dt)*np.random.normal(size=len(living_indices))
    model.Pop[living_indices, 1] += dt*V1 + 0.01*model.Pop[living_indices, 1] \
        * np.sqrt(dt)*np.random.normal(size=len(living_indices))

    # Cell division
    betas = model.beta(model.Pop[living_indices, 1], lambdaS, lambdaD)
    division_indices = living_indices[
        np.random.uniform(size=len(living_indices)) < betas*dt]
    for i, index in enumerate(division_indices):
        if model.Pop[index, 1] > model.S_th:  # Symmetric
            model.Pop[nonliving_indices[i], :] = model.Pop[index, :].copy()
        elif model.Pop[index, 1] >= model.D_th:  # Assymetric
            d_xA = 0.95*model.Pop[index, 0]
            d_xB = 0.95*model.Pop[index, 1]
            model.Pop[nonliving_indices[i], :] = [
                model.Pop[index, 0]+d_xA, model.Pop[index, 1]-d_xB, 1]
            model.Pop[index, :] = [
                model.Pop[index, 0]-d_xA, model.Pop[index, 1]+d_xB, 1]

    living_indices = np.where(model.Pop[:, 2] == 1)[0]

    # Cell death
    death_indices = living_indices[
        np.random.uniform(size=len(living_indices)) < 0.03*dt]
    model.Pop[death_indices, 2] = 0.

model.plot_Pop(living_indices)
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
