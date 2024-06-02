"""
Modifying 2D_model5 so that xB controls cell type

"""


import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from funcs import *

max_cell_num = 100 # Maximum number of cells that can be alive at a time


dt = 0.1
num_steps = 2000
ts = np.arange(0,dt*num_steps,dt)

print('time simulated = ',num_steps*dt)


S_th = 1.5 # xB threshold
D_th = 0.5 # xB threshold

# Initial population:
Pop = np.zeros((max_cell_num,3)) # properties: (xA,xB,alive)
Pop[0,:] = [.1,2.5,1]
Pop[1,:] = [0.2,2.,1]
Pop[2,:] = [0.15,1.5,1]
Pop[3,:] = [.05,.5,1]

living_indices = np.where(Pop[:,2]==1)[0]
nonliving_indices = np.where(Pop[:,2]==0)[0]

plot_Pop(Pop,living_indices,D_th)
#plt.show()
#quit()

#dBs = 1.*np.sqrt(dt)*np.random.normal(size=(max_cell_num,steps))
#print(dBs)
N_Ss = np.zeros(num_steps)
N_Sts = np.zeros(num_steps)
N_Ds = np.zeros(num_steps)
for step in range(num_steps):
	N_Ss[step],N_Sts[step],N_Ds[step] = find_Ns(Pop,S_th,D_th,living_indices=living_indices)
	#print(N_Ss[step],N_Sts[step],N_Ds[step])
	#quit()

	# Lambda dynamics
	lambdaS = np.sum(Pop[living_indices,1]>S_th)
	lambdaD = np.sum(Pop[living_indices,1]<D_th)
	#print(lambdaS,lambdaD)
	#quit()

	#Gene Dynamics
	V0,V1 = V(Pop[living_indices,0],Pop[living_indices,1],lambdaD)
	Pop[living_indices,0] += dt*V0 + 0.01*Pop[living_indices,0]*np.sqrt(dt)*np.random.normal(size=len(living_indices))
	Pop[living_indices,1] += dt*V1 + 0.01*Pop[living_indices,1]*np.sqrt(dt)*np.random.normal(size=len(living_indices))


	#Cell division
	betas = beta(Pop[living_indices,1],lambdaS,lambdaD,S_th,D_th)
	division_indices = living_indices[np.random.uniform(size=len(living_indices)) < betas*dt]
	for i,index in enumerate(division_indices):
		if Pop[index,1]>S_th: # Symmetric
			Pop[nonliving_indices[i],:] = Pop[index,:].copy()
		elif Pop[index,1]>=D_th: # Assymetric
			d_xA = 0.95*Pop[index,0]
			d_xB = 0.95*Pop[index,1]
			Pop[nonliving_indices[i],:] = [Pop[index,0]+d_xA,Pop[index,1]-d_xB,1]
			Pop[index,:] = [Pop[index,0]-d_xA,Pop[index,1]+d_xB,1]

	living_indices = np.where(Pop[:,2]==1)[0]

	# Cell death
	death_indices = living_indices[np.random.uniform(size=len(living_indices)) < 0.03*dt]
	Pop[death_indices,2] = 0.
	living_indices = np.where(Pop[:,2]==1)[0]
	nonliving_indices = np.where(Pop[:,2]==0)[0]


print(Pop)




plot_Pop(Pop,living_indices,D_th)
plt.figure()
plt.plot(ts,N_Ss)
plt.plot(ts,N_Sts)
plt.plot(ts,N_Ds)
plt.plot([-1,ts[-1]+1],[26,26],'k',linewidth=1)
plt.plot([-1,ts[-1]+1],[33,33],'k',linewidth=1)
plt.xlim(-1,ts[-1]+1)
plt.ylim(0,43)
plt.legend(['$N_S$','$N_{S^T}$','$N_D$'],loc='upper left')
plt.gca().spines[['right', 'top']].set_visible(False)
plt.show()
