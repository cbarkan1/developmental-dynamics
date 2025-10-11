"""
Runs simulation of the Example 1 model and plots dynamics (as in Figure 3A)
"""

import numpy as np
import matplotlib.pyplot as plt
from Example_1_Model import Example_1_Model, parameters

dt = 0.02 # Timestep in days
NUM_STEPS = 8000

model = Example_1_Model(parameters, max_cell_num=100)   
model.create_cell(1, 25)
model.create_cell(2, 20.)
model.create_cell(1.5, 15)
model.create_cell(0.5, 5)
ts, N_Ss, N_Sts, N_Ds = model.run_simulation(dt, NUM_STEPS)

plt.figure(figsize=(7,4))
plt.plot(ts, N_Ss)
plt.plot(ts, N_Sts)
plt.plot(ts, N_Ds)
plt.xlim(-1, ts[-1]+1)
plt.ylim(0, 43)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.show()





