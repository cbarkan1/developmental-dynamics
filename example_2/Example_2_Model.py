import numpy as np
import matplotlib.pyplot as plt
import itertools
from voronoi_max_radius import plot_voronoi
cmap = plt.colormaps['viridis']
np.random.seed(0)

parameters = {
    'eps': 1,
    'r_min': 0.08,
    'r0': 1000,
    'r': 2100,
    'K_sqrd': 1200,
    'b': 250,
    'skin_thresh': 8.,
    'divide_thresh': 23,
    'gamma': 20,  # friction coeff
    'beta': 1.,
    'mu': 0.1,
}

class Example_2_Model():
    def __init__(self, parameters, max_cell_num, movement_thresh):
        self.eps = parameters['eps']
        self.r_min = parameters['r_min']
        self.r0 = parameters['r0']
        self.r = parameters['r']
        self.K_sqrd = parameters['K_sqrd']
        self.b = parameters['b']
        self.skin_thresh = parameters['skin_thresh']
        self.divide_thresh = parameters['divide_thresh']
        self.gamma = parameters['gamma']
        self.beta = parameters['beta']
        self.mu = parameters['mu']
        self.max_cell_num = max_cell_num
        self.movement_thresh = movement_thresh
        self.Pop = np.zeros((self.max_cell_num, 4))  # (r1, r2, x, is_alive)
        
    def F(self,x, c1): # Waddington's landscape
        # units: nM
        return self.r0 + self.r*c1*x**2/(self.K_sqrd + x**2) - self.b*x

    def force(self,r):
        return 2*self.eps*self.r_min**2/r**3 - 2*self.eps*self.r_min/r**2

    def pair_force_vector(self,pair):
        r_vect = self.Pop[pair[0], 0:2] - self.Pop[pair[1], 0:2]
        r = (r_vect[0]**2 + r_vect[1]**2 + 1e-6)**0.5
        f = self.force(r)
        Fx = f*r_vect[0]/r
        Fy = f*r_vect[1]/r
        return Fx, Fy

    def find_forces(self, living_indices):
        net_forces = np.zeros((self.max_cell_num, 2))
        for pair in itertools.combinations(living_indices, 2):
            net_forces[pair[0], :] += self.pair_force_vector(pair)
            net_forces[pair[1], :] -= self.pair_force_vector(pair)

        return net_forces

    def find_cs(self, x, y, living_indices): 
        def c1_tilde(x, y): # TODO: pull out parametes like .07
            return np.exp(-.5*(x**2+y**2)/.07**2)

        def c2_tilde(x, y):
            return np.exp(-.5*(x**2+y**2)/.3**2)

        c1 = np.sum(c1_tilde(x-self.Pop[living_indices, 0], y-self.Pop[living_indices, 1]))
        c2 = np.sum(c2_tilde(x-self.Pop[living_indices, 0], y-self.Pop[living_indices, 1]))

        return c1, c2

    def check_forces_and_cell_speed(self, dt):
        rs = np.linspace(1e-8,1,1000)
        fs = self.force(rs)
        max_attraction_force = abs(np.nanmin(fs))
        max_speed = max_attraction_force/self.gamma
        assert self.movement_thresh >= 2*max_speed*dt, 'dt is too large because movement_thresh < 2*max_speed*dt'

    def run_simulation(self, dt, num_steps, save_dt):
        
        # Ensure that dt is small enough:
        self.check_forces_and_cell_speed(dt)

        ts = np.arange(0, dt*num_steps, dt)
        total_pop = np.zeros(num_steps)
        interior_pop = np.zeros(num_steps)
        dividing_pop = np.zeros(num_steps)
        total_pop[0] = np.sum(self.Pop[:,3])
        interior_pop[0] = np.nan
        dividing_pop[0] = np.nan

        t_record = np.arange(0,dt*num_steps,save_dt)
        Pop_record = np.zeros((len(t_record), self.max_cell_num, 4))
        Pop_record[0,:,:] = self.Pop[:,:]

        for step in range(1,num_steps):
            living_indices = np.where(self.Pop[:, 3] == 1)[0]
            nonliving_indices = np.where(self.Pop[:, 3] == 0)[0]
            if len(nonliving_indices) == 0:
                print('WARNING: reached max_cell_num')

            self.Pop[:, 0:2] += np.clip(dt*self.find_forces(living_indices)/self.gamma, a_min=-self.movement_thresh, a_max=self.movement_thresh)
            self.Pop[:, 0:2] = np.clip(self.Pop[:, 0:2],a_min=0, a_max=1) # Imposing boundaries

            for cell in living_indices:
                r1, r2 = self.Pop[cell, 0:2]
                c1, c2 = self.find_cs(r1, r2, living_indices)

                # gene dynamics
                self.Pop[cell, 2] += np.clip(dt*self.F(self.Pop[cell, 2], c1), a_min=-1, a_max=1) + 0.1*self.Pop[cell, 2]*dt**0.5*np.random.normal()

                # cell division
                if c2 < self.divide_thresh and len(nonliving_indices) > 0 and np.random.rand() < dt*self.beta:
                    dividing_pop[step] += 1
                    shift_r1, shift_r2 = 0.05*np.random.rand(), 0.05*np.random.rand()
                    self.Pop[nonliving_indices[0], :] = (r1+shift_r1)%1, (r2+shift_r2)%1, self.Pop[cell, 2], 1
                    nonliving_indices = nonliving_indices[1:]

            # cell death
            deaths = np.random.binomial(n=1, p=self.mu*dt, size = len(living_indices)).astype(bool)
            self.Pop[living_indices[deaths], 3] = 0

            total_pop[step] = np.sum(self.Pop[:,3])

            if step % (save_dt/dt) == 0:
                record_index = int(step/(save_dt/dt))
                print('step',step, '  record_index',record_index)
                Pop_record[record_index,:,:] = self.Pop.copy()

        return ts, total_pop, Pop_record
    
    def cut_organism_in_half(self):
        living_indices = np.where(self.Pop[:, 3] == 1)[0]
        center_x = np.sum(self.Pop[living_indices,0])/len(living_indices)
        for cell in living_indices:
            if self.Pop[cell,0] > center_x:
                self.Pop[cell,3] = 0
        return self.Pop.copy()

    def plot_Pop(self):
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(10,8))

        living_indices = np.where(self.Pop[:, 3] == 1)[0]

        plot_voronoi(ax0, self.Pop[living_indices,0:2],0.05,(0,1),(0,1),cell_colors=cmap(self.Pop[living_indices,2]/30))

        for cell in living_indices:
            r1, r2, x = self.Pop[cell, 0:3]
            ax1.plot(r1, r2, 'o', color=cmap(x))
            ax2.plot(r1, r2, 'o', color=cmap(x))
        
        grid_points = 50
        xs = np.linspace(0, 1, grid_points)
        ys = np.linspace(0, 1, grid_points)
        c1s = np.zeros((grid_points,grid_points))
        c2s = np.zeros((grid_points,grid_points))
        for i in range(grid_points):
            for ii in range(grid_points):
                c1s[i, ii], c2s[i, ii] = self.find_cs(xs[i], ys[ii], living_indices)

        x_mesh, y_mesh = np.meshgrid(xs,ys)

        cont1 = ax1.contour(x_mesh.T, y_mesh.T, c1s)
        ax1.clabel(cont1, cont1.levels, inline=True, fontsize=10)

        cont2 = ax2.contour(x_mesh.T, y_mesh.T, c2s)
        ax2.clabel(cont2, cont2.levels, inline=True, fontsize=10)

        ax1.set_title('total pop='+str(np.sum(self.Pop[:,3])))
        ax1.set_title('total pop='+str(np.sum(self.Pop[:,3])))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax3.hist(self.Pop[living_indices,2])
        ax3.set_title("Distribution of cell state")
        plt.show()
        return