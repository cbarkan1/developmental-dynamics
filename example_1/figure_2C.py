import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Example_1_Model import Example_1_Model, parameters


def dYdt(Y, t):
    F1, F2 = model.F(Y[0], Y[1], c_1=lambdaD)
    return [F1, F2]

def jacobian(Y,lambdaD):
    F1, F2 = model.F(Y[0], Y[1], c_1=lambdaD)
    eps = 1e-8
    F1_1, F2_1 = model.F(Y[0]+eps, Y[1], c_1=lambdaD)
    F1_2, F2_2 = model.F(Y[0], Y[1]+eps, c_1=lambdaD)
    d1F1 = (F1_1 - F1)/eps
    d2F1 = (F1_2 - F1)/eps
    d1F2 = (F2_1 - F2)/eps
    d2F2 = (F2_2 - F2)/eps
    return np.array([[d1F1, d1F2], [d2F1, d2F2]]), np.array([F1, F2])

def find_nearby_fixed_point(Y0, lambdaD, num_iter=200, alpha=0.3):
    for i in range(num_iter):
        J, dF = jacobian(Y0, lambdaD)
        print(Y0, dF)
        raw_step = np.linalg.solve(J,dF)
        raw_step_norm = (raw_step[0]**2 + raw_step[1]**2)**0.5 #np.linalg.norm(raw_step)
        step_norm = min(raw_step_norm,0.1)
        step = raw_step*step_norm/raw_step_norm
        Y0 = Y0 - alpha*step
    return Y0

def get_sadde1s(c_1s):
    Y0 = np.array([27,4.8])
    saddle1_0 = find_nearby_fixed_point(Y0, 0)

    num_pts = len(c_1s)
    saddle1s = np.zeros((num_pts,2))
    saddle1s[0,:] = saddle1_0[:]
    for i in range(1,num_pts):
        c_1 = c_1s[i]
        saddle1s[i,:] = find_nearby_fixed_point(saddle1s[i-1,:], c_1, num_iter=100,alpha=0.1)[:]
    return saddle1s

def get_sadde2s():
    c_1s = np.linspace(30,34,20)
    c_1s_down = np.linspace(30,25.7,20)
    Y0 = np.array([4.48,15.9])
    saddle2_0 = find_nearby_fixed_point(Y0, c_1s[0], num_iter=400, alpha=0.1)

    num_pts = len(c_1s)
    saddle2s = np.zeros((num_pts,2))
    saddle2s[0,:] = saddle2_0[:]
    saddle2s_down = np.zeros((num_pts,2))
    saddle2s_down[0,:] = saddle2_0[:]
    for i in range(1,num_pts):
        lambdaD = c_1s[i]
        lambdaD_down = c_1s_down[i]
        saddle2s[i,:] = find_nearby_fixed_point(saddle2s[i-1,:], lambdaD, num_iter=100,alpha=0.1)[:]
        saddle2s_down[i,:] = find_nearby_fixed_point(saddle2s_down[i-1,:], lambdaD_down, num_iter=100,alpha=0.1)[:]

    c_1s = np.concatenate((c_1s,c_1s_down))
    saddle2s = np.concatenate((saddle2s, saddle2s_down))
    return saddle2s, c_1s

model = Example_1_Model(parameters, max_cell_num=1)

c_1s = np.linspace(0, 50, 400)
saddle1s = get_sadde1s(c_1s)
saddle2s, saddle2_c_1s = get_sadde2s()

Y_attractors = np.zeros((3, len(c_1s), 2))
for i in range(1,len(c_1s)):
    lambdaD = c_1s[i]
    Y1s = [[-10, 0], [20, 0], [30, 30]] # Order: S, D, ST
    for ii, Y1 in enumerate(Y1s):
        Y1_of_t = odeint(dYdt, Y1, np.linspace(0, 100, 10))
        Y_attractors[ii, i, :] = Y1_of_t[-1, :]

bottom_break = 203
top_break = 270
a_S = Y_attractors[0,bottom_break:,:]
a_ST = np.concatenate((Y_attractors[0, 0:bottom_break, :], Y_attractors[2, bottom_break:top_break, :]))
a_D = Y_attractors[1,:,:]

saddle2s_0 = np.concatenate((saddle2s[18::-1,0],saddle2s[20:,0]))
saddle2s_1 = np.concatenate((saddle2s[18::-1,1],saddle2s[20:,1]))
c_1_2s = np.concatenate((saddle2_c_1s[18::-1],saddle2_c_1s[20:]))
top_point_0, top_point_1  = (a_ST[-1,0] + saddle2s_0[0])/2, (a_ST[-1,1] + saddle2s_1[0])/2
top_point_3 = c_1s[top_break+1]
lambdaDs_ST = np.append(c_1s[:len(a_ST[:,0])],top_point_3)
a_ST = np.concatenate((a_ST,np.array([[top_point_0, top_point_1]])))

bottom_point_0, bottom_point_1  = (a_S[0,0] + saddle2s_0[-1])/2, (a_S[0,1] + saddle2s_1[-1])/2
bottom_point_3 = c_1s[bottom_break]
a_S = np.concatenate((np.array([[bottom_point_0, bottom_point_1]]), a_S))

saddle2s_0 = np.insert(saddle2s_0, 0, top_point_0)
saddle2s_1 = np.insert(saddle2s_1, 0, top_point_1)
saddle2s_0 = np.append(saddle2s_0, bottom_point_0)
saddle2s_1 = np.append(saddle2s_1, bottom_point_1)
c_1_2s = np.insert(c_1_2s, 0, top_point_3)
c_1_2s = np.append(c_1_2s, bottom_point_3)

ax = plt.figure().add_subplot(projection='3d')
linewidth=2
ax.plot(a_S[:,0], a_S[:,1], [c_1s[bottom_break], *c_1s[bottom_break:]],color='k',linewidth=linewidth)
ax.plot(a_D[1:,0], a_D[1:,1], c_1s[1:],color='k',linewidth=linewidth)
ax.plot(saddle1s[:,0], saddle1s[:,1], c_1s,color='red',linewidth=linewidth)
ax.plot(saddle2s_0, saddle2s_1, c_1_2s,color='red',linewidth=linewidth)
ax.plot(a_ST[1:,0], a_ST[1:,1], lambdaDs_ST[1:],color='k',linewidth=linewidth)
ax.elev = 10
ax.azim = 260
ax.set_xticklabels([''])
ax.set_yticklabels([''])
ax.set_zticklabels([''])
plt.show()
