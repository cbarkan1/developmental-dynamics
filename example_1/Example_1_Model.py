"""
Defines Example_1_Model class, which contains all methods needed
for simulating the dynamics.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

parameters = {
    'a1': 1,  # Parameter in Equation 20, units: 1/day
    'a2': 0.15,  # Parameter in Equation 20, unitless
    'a3': 0.5,  # Parameter in Equation 20, units: 1/nM
    'beta_Sstar': 0.6,  # Triggered stem cell division rate, units: 1/day
    'w': 0.95,  # Coefficient on w(z) vector, unitless
    'mu': 0.02,  # Death rate, units: 1/day
    'L_over_d': 1, # Ratios L_1/d_1 and L_2/d_2 in Equation 23. units: nM
    'zS': 15,  # Threshold value of z_B, separating stem cells (S) from triggered stem cells (S*), units: nM
    'zD': 5,  # Threshold value of z_B, separating differentiated cells (D) from triggered stem cells (S*), units: nM
    'r_SA0': 250,  # Rate r_{S,A}(0), units: nM/day
    'gamma': 20,  # Constant in Equation 19, units: nM/day
    'r_0A': 500,  # Rate r_{0,A}, units: nM/day
    'r_0B': 500,  # Rate r_{0,B}, units: nM/day
    'r_ESA0': 1000,  # Rate r_{ES,A}(0), units: nM/day
    'r_EA': 1500,  # Rate r_{E,A}, units: nM/day
    'r_EB': 1000,  # Rate r_{E,B}, units: nM/day
    'k_A': 50,  # Degradation rate k_A, units: 1/day
    'k_B': 50,  # Degradation rate k_B, units: 1/day
    'r_SB': 0,  # Rate r_{S,B}, units: nM/day
    'r_ESB': 500,  # Rate r_{ES,B}, units: nM/day
    'sigma': 0.1,  # Coefficient on sigma(z;c), units: 1/(day^{1/2}*nM)
    'K': 5, # K_{E,m} and K_{S,m} for m=A,B. units: nM
}

class Example_1_Model:
    def __init__(self, parameters: dict, max_cell_num: int):
        self.a1 = parameters['a1'] # Parameter in Equation 20, units: 1/day
        self.a2 = parameters['a2'] # Parameter in Equation 20, unitless
        self.a3 = parameters['a3'] # Parameter in Equation 20, units: 1/nM
        self.beta_Sstar = parameters['beta_Sstar'] # Triggered stem cell division rate, units: 1/day
        self.w = parameters['w'] # Coefficient on w(z) vector, unitless
        self.mu = parameters['mu'] # Death rate, units: 1/day
        self.L_over_d = parameters['L_over_d']
        self.zS = parameters['zS'] # Threshold value of z_B, separating stem cells (S) from triggered stem cells (S*), units: nM
        self.zD = parameters['zD'] # Threshold value of z_B, separating differentiated cells (D) from triggered stem cells (S*), units: nM
        self.r_SA0 = parameters['r_SA0'] # Rate r_{S,A}(0), units: nM/day
        self.gamma = parameters['gamma'] # Constant in Equation 19, units: nM/day
        self.r_0A = parameters['r_0A'] # Rate r_{0,A}, units: nM/day
        self.r_0B = parameters['r_0B'] # Rate r_{0,B}, units: nM/day
        self.r_ESA0 = parameters['r_ESA0'] # Rate r_{ES,A}(0), units: nM/day
        self.r_EA = parameters['r_EA'] # Rate r_{E,A}, units: nM/day
        self.r_EB = parameters['r_EB'] # Rate r_{E,B}, units: nM/day
        self.k_A = parameters['k_A'] # Degradation rate k_A, units: 1/day
        self.k_B = parameters['k_B'] # Degradation rate k_B, units: 1/day
        self.r_SB = parameters['r_SB'] # Rate r_{S,B}, units: nM/day
        self.r_ESB = parameters['r_ESB'] # Rate r_{ES,B}, units: nM/day
        self.sigma = parameters['sigma'] # Coefficient on sigma(z;c), units: 1/(day^{1/2}*nM)

        self.K4 = parameters['K']**4 # K^4 where K is K_{E,m} and K_{S,m} for m=A,B. units: nM
        self.Pop = np.zeros((max_cell_num, 3))  # properties: (xA,xB,alive)

    def create_cell(self, z_A, z_B):
        """ Add a cell in state (z_A, z_B) to the population."""
        unliving_indices = np.where(self.Pop[:, 2] == 0)[0]
        self.Pop[unliving_indices[0], :] = [z_A, z_B, 1]

    def F(self, z_A, z_B, c_1):
        """ Waddington Vector Field"""
        P_PA = z_A**4/(self.K4+z_A**4)
        P_PB = z_B**4/(self.K4+z_B**4)
        P_OA = P_PB
        P_OB = P_PA
        frac_BD = self.gamma/(self.gamma + c_1) # Fraction of B tetramer bound to c_1 molecules

        r_0A = self.r_0A
        r_PA = self.r_EA
        r_OA = self.r_SA0 * frac_BD
        r_POA = self.r_ESA0 * frac_BD

        F_A = r_0A*(1-P_PA)*(1-P_OA) + r_PA*P_PA*(1-P_OA) \
            + r_OA*(1-P_PA)*P_OA + r_POA*P_PA*P_OA - self.k_A*z_A

        r_0B = self.r_0B
        r_PB = self.r_EB
        r_OB = self.r_SB
        r_POB = self.r_ESB

        F_B = r_0B*(1-P_PB)*(1-P_OB) + r_PB*P_PB*(1-P_OB) \
            + r_OB*(1-P_PB)*P_OB + r_POB*P_PB*P_OB - self.k_B*z_B

        return F_A, F_B

    def update_gene_state(self, dt, living_indices, c_1):
        V0, V1 = self.F(self.Pop[living_indices, 0],
                        self.Pop[living_indices, 1], c_1)
        self.Pop[living_indices, 0] += dt*V0 \
            + self.sigma*self.Pop[living_indices, 0] \
            * np.sqrt(dt)*np.random.normal(size=len(living_indices))
        self.Pop[living_indices, 1] += dt*V1 \
            + self.sigma*self.Pop[living_indices, 1] \
            * np.sqrt(dt)*np.random.normal(size=len(living_indices))

    def beta(self, xB, c_2, c_1):
        """
        Division rate
        """
        return np.heaviside(xB-self.zS, 1) * self.a1/(1. + self.a2*np.exp(c_2*self.a3)) \
            + self.beta_Sstar * np.heaviside(self.zS-xB, 1)*np.heaviside(xB-self.zD, 1)

    def find_cell_type_numbers(self, living_indices=None):
        if living_indices is None:
            living_indices = np.where(self.Pop[:, 2] == 1)[0]

        N_S = np.sum(self.Pop[living_indices, 1] > self.zS)
        N_St = np.sum(np.logical_and(self.Pop[living_indices, 1] <= self.zS,
                                     self.Pop[living_indices, 1] >= self.zD))
        N_D = np.sum(self.Pop[living_indices, 1] < self.zD)
        return N_S, N_St, N_D

    def find_signaling_molecule_concentrations(self, living_indices):
        """
        Calculates concentration of signaling molecules
        """
        c_1 = self.L_over_d*np.sum(self.Pop[living_indices, 1] < self.zD)
        c_2 = self.L_over_d*np.sum(self.Pop[living_indices, 1] > self.zS)
        return c_1, c_2

    def cell_division(self, dt, living_indices, nonliving_indices,
                      c_1, c_2):
        betas = self.beta(self.Pop[living_indices, 1], c_2, c_1)
        division_indices = living_indices[
            np.random.uniform(size=len(living_indices)) < betas*dt]
        for i, index in enumerate(division_indices):
            if self.Pop[index, 1] > self.zS:  # Symmetric
                self.Pop[nonliving_indices[i], :] = self.Pop[index, :].copy()
            elif self.Pop[index, 1] >= self.zD:  # Assymetric
                d_xA = self.w*self.Pop[index, 0]
                d_xB = self.w*self.Pop[index, 1]
                self.Pop[nonliving_indices[i], :] = [
                    self.Pop[index, 0]+d_xA, self.Pop[index, 1]-d_xB, 1]
                self.Pop[index, :] = [
                    self.Pop[index, 0]-d_xA, self.Pop[index, 1]+d_xB, 1]

    def cell_death(self, dt, living_indices):
        death_indices = living_indices[
            np.random.uniform(size=len(living_indices)) < self.mu*dt]
        self.Pop[death_indices, 2] = 0

    def run_simulation(self, dt, num_steps):
        ts = np.arange(0, dt*num_steps, dt)
        N_Ss = np.zeros(num_steps)
        N_Sts = np.zeros(num_steps)
        N_Ds = np.zeros(num_steps)
        for step in range(num_steps):
            # Check which indices correspond to currently living cells:
            living_indices = np.where(self.Pop[:, 2] == 1)[0]
            nonliving_indices = np.where(self.Pop[:, 2] == 0)[0]

            # Record populations
            N_Ss[step], N_Sts[step], N_Ds[step] = self.find_cell_type_numbers(living_indices)

            c_1, c_2 = self.find_signaling_molecule_concentrations(living_indices)
            self.update_gene_state(dt, living_indices, c_1)
            self.cell_division(dt, living_indices, nonliving_indices, c_1, c_2)
            self.cell_death(dt, living_indices)

        return ts, N_Ss, N_Sts, N_Ds

    def plot_population_on_Waddington_vectorfield(self, living_indices=None, show=True):
        """
        Plots the cell state of each cell on the Waddington vector field
        """
        if living_indices is None:
            living_indices = np.where(self.Pop[:, 2] == 1)[0]

        c_1 = np.sum(self.Pop[living_indices, 1] < self.zD)

        def dYdt(Y, t):
            V1, V2 = self.F(Y[0], Y[1], c_1=c_1)
            return [V1, V2]

        plt.figure()
        y1_range = np.linspace(0, 35, 50)
        y2_range = np.linspace(0, 35, 50)
        y1_mesh, y2_mesh = np.meshgrid(y1_range, y2_range)
        V1, V2 = self.F(y1_mesh, y2_mesh, c_1=c_1)

        plt.streamplot(y1_mesh, y2_mesh, V1, V2)

        Y1s = [[-10, 0], [20, 0], [30, 30]]
        for Y1 in Y1s:
            Y1_of_t = odeint(dYdt, Y1, np.linspace(0, 200, 10))
            plt.plot(Y1_of_t[-1, 0], Y1_of_t[-1, 1],
                     's', color='k', markersize=10)

        for index in living_indices:
            plt.plot(self.Pop[index, 0], self.Pop[index, 1], 'o')

        plt.xlabel('Genes')
        plt.ylabel('Spatial organization')
        plt.title('c_1 = '+str(c_1))

        if show:
            plt.show()

        return
