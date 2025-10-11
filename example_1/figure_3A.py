"""
Generates Figure 3A from saved data of 100 simulation trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

file = np.load('figure_3A_data.npz')
ts = file['ts']
S = file['S']
St = file['St']
D = file['D']
N = file['N']
alpha = 0.02

plt.figure(figsize=(7,4))

# Plot all trajectories with translucent color.
for n in range(N):
    plt.plot(ts, S[n,:], color=colors[0], alpha=alpha)
    plt.plot(ts, St[n,:], color=colors[1], alpha=alpha)
    plt.plot(ts, D[n,:], color=colors[2], alpha=alpha)

# Plot one trajectory with full color.
n = 0
plt.plot(ts, S[n,:], color=colors[0])
plt.plot(ts, St[n,:], color=colors[1])
plt.plot(ts, D[n,:], color=colors[2])
plt.xlim(-1, 110)
plt.ylim(0, 43)

ax = plt.gca()
ax.set_xticks([0,10,20,30,40,50,60,70,80,90,100,110])
ax.set_xticklabels([0,'',20,'',40,'',60,'',80,'',100,''])

plt.gca().spines[['right', 'top']].set_visible(False)
plt.show()