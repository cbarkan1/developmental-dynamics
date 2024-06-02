import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def plot_Pop(Pop,living_indices,D_th):
	lambdaD = np.sum(Pop[living_indices,1]<D_th)

	def dYdt(Y,t):
		V1,V2 = V(Y[0],Y[1],lambdaD=lambdaD)
		return [V1,V2]

	plt.figure()
	y1_range = np.linspace(0,3.5,50)
	y2_range = np.linspace(0,3.5,50)
	y1_mesh,y2_mesh = np.meshgrid(y1_range,y2_range)
	V1,V2 = V(y1_mesh,y2_mesh,lambdaD=lambdaD)

	#plt.quiver(y1_mesh,y2_mesh,np.tanh(V1/3),np.tanh(V2/3))
	plt.streamplot(y1_mesh,y2_mesh,V1,V2)

	Y1s = [[-1,0],[2,0],[3,3]]
	for Y1 in Y1s:
		Y1_of_t = odeint(dYdt,Y1,np.linspace(0,200,10))
		plt.plot(Y1_of_t[-1,0],Y1_of_t[-1,1],'s',color='k',markersize=10)


	for index in living_indices:
		plt.plot(Pop[index,0],Pop[index,1],'o')

	plt.xlabel('Genes')
	plt.ylabel('Spatial organization')
	plt.title('lambdaD = '+str(lambdaD))
	return

def find_Ns(Pop,S_th,D_th,living_indices=None):
	if living_indices is None:
		living_indices = np.where(Pop[:,2]==1)[0]

	N_S = np.sum( Pop[living_indices,1]>S_th )
	N_St = np.sum( np.logical_and(Pop[living_indices,1]<=S_th , Pop[living_indices,1]>=D_th))
	N_D  = np.sum( Pop[living_indices,1]<D_th )
	return N_S,N_St,N_D


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


r_R1 = 0.5 #* 0.4
alpha1 = 20
r_01 = 1
r_Rt1 = 2 #* .4 #1
r_t1 = 3 # Expression rate with only t.f. bound.
k1 = 1


r_R2 = 0
r_02 = 1
r_Rt2 = 1
r_t2 = 2 # Expression rate with only t.f. bound.
k2 = 1

n = 4
a = .5**n


def V(y1,y2,lambdaD):
	Pt1 = y1**n/(a+y1**n)
	GammaR1 = a/(a+y2**n)
	Gamma_D1 = alpha1/(alpha1 + lambdaD)
	#V1 = r_R1 + Pt1*(r_Rt1-r_R1) + GammaR1*(r_01-r_R1) + Pt1*GammaR1*(r_t1+r_R1-r_01-r_Rt1) - k1*y1
	V1 = r_R1*Gamma_D1 + Pt1*(r_Rt1-r_R1)*Gamma_D1 + GammaR1*(r_01-r_R1*Gamma_D1) + Pt1*GammaR1*(r_t1+r_R1*Gamma_D1-r_01-r_Rt1*Gamma_D1) - k1*y1


	Pt2 = y2**n/(a+y2**n)
	GammaR2 = a/(a+y1**n)
	V2 = r_R2 + Pt2*(r_Rt2-r_R2) + GammaR2*(r_02-r_R2) + Pt2*GammaR2*(r_t2+r_R2-r_02-r_Rt2) - k2*y2

	return V1 , V2

def beta(xB,lambdaS,lambdaD,S_th,D_th):
	return  np.heaviside(xB - S_th,1) * 7./(7. + np.exp(lambdaS*0.5)) + \
			np.heaviside(S_th - xB,1)*np.heaviside(xB - D_th,1) * 0.6



if 0: # Plot V
	y1_range = np.linspace(0,4,100)
	y2_range = np.linspace(0,3,100)
	y1_mesh,y2_mesh = np.meshgrid(y1_range,y2_range)
	V1,V2 = V(y1_mesh,y2_mesh,34)
	plt.streamplot(y1_mesh,y2_mesh,V1,V2)
	plt.gca().set_aspect('equal')
	plt.show()


