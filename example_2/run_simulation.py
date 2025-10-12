"""
Runs simulation of the Example 2 model and plots the populations.
"""

import numpy as np
import matplotlib.pyplot as plt
from Example_2_Model import Example_2_Model, parameters
np.random.seed(0)

max_cell_num = 50
movement_thresh = 0.001
model = Example_2_Model(parameters, max_cell_num, movement_thresh)


# Create initial population
spacing = 0.07
cell_count = 0
for i in range(1):
    for ii in range(1):
        # TODO: Make a create_cell method
        model.Pop[cell_count, :] = 0.4 + i*spacing, 0.4 + ii*spacing, 0, 1
        cell_count += 1


dt = 0.002 # 0.002 days is about 3 minutes
num_steps = 4000
save_dt = .1 # time between saves
ts1, total_pop1, Pop_record1 = model.run_simulation(dt, num_steps//2, save_dt)
model.cut_organism_in_half()
ts2, total_pop2, Pop_record2 = model.run_simulation(dt, num_steps//2, save_dt)
total_pop = np.concatenate((total_pop1, total_pop2))
ts = np.concatenate((ts1, ts2+ts1[-1]))
Pop_record = np.concatenate((Pop_record1, Pop_record2), axis=0)
model.plot_Pop()

#np.savez('model6_sim1.npz', t_record=ts, Pop_record=Pop_record)

plt.plot(ts, total_pop, label='total')
plt.xlabel('time (days)')
plt.ylabel('total population')
plt.show()


