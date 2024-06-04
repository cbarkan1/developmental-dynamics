import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class Model:

    def __init__(self, S_th, D_th, r_R1, alpha1, r_01, r_Rt1, 
                 r_t1, k1, r_R2, r_02, r_Rt2, r_t2, k2, n, a, sigma):

        self.S_th = S_th
        self.D_th = D_th
        self.r_R1 = r_R1
        self.alpha1 = alpha1
        self.r_01 = r_01
        self.r_Rt1 = r_Rt1
        self.r_t1 = r_t1
        
        self.k1 = k1
        self.k2 = k2
        self.r_R2 = r_R2
        self.r_02 = r_02
        self.r_Rt2 = r_Rt2
        self.r_t2 = r_t2

        self.n = n
        self.a = a
        self.sigma = sigma
        self.Pop = None

    def create_Pop(self, max_cell_num):
        self.Pop = np.zeros((max_cell_num, 3))  # properties: (xA,xB,alive)

    def create_cell(self, xA, xB):
        unliving_indices = np.where(self.Pop[:, 2] == 0)[0]
        self.Pop[unliving_indices[0], :] = [xA, xB, 1]

    def V(self, y1, y2, lambdaD):
        Pt1 = y1**self.n/(self.a+y1**self.n)
        GammaR1 = self.a/(self.a+y2**self.n)
        Gamma_D1 = self.alpha1/(self.alpha1 + lambdaD)
        # V1 = r_R1 + Pt1*(r_Rt1-r_R1) + GammaR1*(r_01-r_R1) + \
        #     Pt1*GammaR1*(r_t1+r_R1-r_01-r_Rt1) - k1*y1
        V1 = self.r_R1*Gamma_D1 + Pt1*(self.r_Rt1-self.r_R1)*Gamma_D1 \
            + GammaR1*(self.r_01-self.r_R1*Gamma_D1) \
            + Pt1*GammaR1*(self.r_t1+self.r_R1*Gamma_D1
                           - self.r_01-self.r_Rt1*Gamma_D1) - self.k1*y1

        Pt2 = y2**self.n/(self.a+y2**self.n)
        GammaR2 = self.a/(self.a+y1**self.n)
        V2 = self.r_R2 + Pt2*(self.r_Rt2-self.r_R2) \
            + GammaR2*(self.r_02-self.r_R2) \
            + Pt2*GammaR2*(self.r_t2+self.r_R2-self.r_02-self.r_Rt2) \
            - self.k2*y2

        return V1, V2

    def update_gene_state(self, dt, living_indices, lambdaD):
        V0, V1 = self.V(self.Pop[living_indices, 0],
                        self.Pop[living_indices, 1], lambdaD)
        self.Pop[living_indices, 0] += dt*V0 \
            + self.sigma*self.Pop[living_indices, 0] \
            * np.sqrt(dt)*np.random.normal(size=len(living_indices))
        self.Pop[living_indices, 1] += dt*V1 \
            + self.sigma*self.Pop[living_indices, 1] \
            * np.sqrt(dt)*np.random.normal(size=len(living_indices))

    def beta(self, xB, lambdaS, lambdaD):
        return np.heaviside(xB-self.S_th, 1) * 7./(7. + np.exp(lambdaS*0.5)) \
            + 0.6 * np.heaviside(self.S_th-xB, 1)*np.heaviside(xB-self.D_th, 1)

    def find_Ns(self, living_indices=None):
        if living_indices is None:
            living_indices = np.where(self.Pop[:, 2] == 1)[0]

        N_S = np.sum(self.Pop[living_indices, 1] > self.S_th)
        N_St = np.sum(np.logical_and(self.Pop[living_indices, 1] <= self.S_th,
                                     self.Pop[living_indices, 1] >= self.D_th))
        N_D = np.sum(self.Pop[living_indices, 1] < self.D_th)
        return N_S, N_St, N_D

    def find_lambdas(self, living_indices):
        lambdaS = np.sum(self.Pop[living_indices, 1] > self.S_th)
        lambdaD = np.sum(self.Pop[living_indices, 1] < self.D_th)
        return lambdaS, lambdaD

    def cell_division(self, dt, living_indices, nonliving_indices,
                      lambdaS, lambdaD):
        betas = self.beta(self.Pop[living_indices, 1], lambdaS, lambdaD)
        division_indices = living_indices[
            np.random.uniform(size=len(living_indices)) < betas*dt]
        for i, index in enumerate(division_indices):
            if self.Pop[index, 1] > self.S_th:  # Symmetric
                self.Pop[nonliving_indices[i], :] = self.Pop[index, :].copy()
            elif self.Pop[index, 1] >= self.D_th:  # Assymetric
                d_xA = 0.95*self.Pop[index, 0]
                d_xB = 0.95*self.Pop[index, 1]
                self.Pop[nonliving_indices[i], :] = [
                    self.Pop[index, 0]+d_xA, self.Pop[index, 1]-d_xB, 1]
                self.Pop[index, :] = [
                    self.Pop[index, 0]-d_xA, self.Pop[index, 1]+d_xB, 1]

    def cell_death(self, dt, living_indices):
        death_indices = living_indices[
            np.random.uniform(size=len(living_indices)) < 0.03*dt]
        self.Pop[death_indices, 2] = 0.

    def plot_Pop(self, living_indices=None):
        if living_indices is None:
            living_indices = np.where(self.Pop[:, 2] == 1)[0]

        lambdaD = np.sum(self.Pop[living_indices, 1] < self.D_th)

        def dYdt(Y, t):
            V1, V2 = self.V(Y[0], Y[1], lambdaD=lambdaD)
            return [V1, V2]

        plt.figure()
        y1_range = np.linspace(0, 3.5, 50)
        y2_range = np.linspace(0, 3.5, 50)
        y1_mesh, y2_mesh = np.meshgrid(y1_range, y2_range)
        V1, V2 = self.V(y1_mesh, y2_mesh, lambdaD=lambdaD)

        plt.streamplot(y1_mesh, y2_mesh, V1, V2)

        Y1s = [[-1, 0], [2, 0], [3, 3]]
        for Y1 in Y1s:
            Y1_of_t = odeint(dYdt, Y1, np.linspace(0, 200, 10))
            plt.plot(Y1_of_t[-1, 0], Y1_of_t[-1, 1],
                     's', color='k', markersize=10)

        for index in living_indices:
            plt.plot(self.Pop[index, 0], self.Pop[index, 1], 'o')

        plt.xlabel('Genes')
        plt.ylabel('Spatial organization')
        plt.title('lambdaD = '+str(lambdaD))
        return
