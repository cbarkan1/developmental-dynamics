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
    # Update living_indices and nonliving_indices
    living_indices = np.where(model.Pop[:, 2] == 1)[0]
    nonliving_indices = np.where(model.Pop[:, 2] == 0)[0]

    # Record populations
    N_Ss[step], N_Sts[step], N_Ds[step] = model.find_Ns(
        living_indices=living_indices)

    # Lambda dynamics
    lambdaS, lambdaD = model.find_lambdas(living_indices)

    # Gene regulatory dynamics
    model.update_gene_state(dt, living_indices, lambdaD)

    # Cell division
    model.cell_division(dt, living_indices, nonliving_indices,
                        lambdaS, lambdaD)

    # Cell death
    model.cell_death(dt, living_indices)

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
