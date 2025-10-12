import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from voronoi_max_radius import plot_voronoi


def figure_with_cs(indices):
    # Plot the organism and the concentrations

    res = 300
    x_mesh, y_mesh = np.meshgrid(np.linspace(0,1,res), np.linspace(0,1,res))

    fig, ax = plt.subplots(3,len(indices), figsize=(14,7.1))
    xlim = (0.1,.7)
    ylim = (0.15,.75)
    for i, index in enumerate(indices):
        living_indices = np.where(Pops[index,:, 3] == 1)[0]
        Pop_living = Pops[index,living_indices,:]
        plot_voronoi(ax[0,i], Pop_living[:,0:2],0.05,xlim,ylim,cell_colors=cmap(Pop_living[:,2]/30),point_size=2, boundary_width=1)
        ax[0,i].set_title(f"t = {ts[index]:.1f} days")

        c1s, c2s = find_cs_vectorized(x_mesh, y_mesh, Pop_living)
        #ax[1,i].contour(x_mesh.T, y_mesh.T, c1s.T, levels=[0,2,4,6,8,10,12,14,16])
        ax[1,i].pcolormesh(x_mesh, y_mesh, c1s, vmin=0, vmax=12, cmap='Blues', rasterized=True) 
        ax[1,i].plot(Pop_living[:,0], Pop_living[:,1],'.', color='k')

        #ax[2,i].contour(x_mesh.T, y_mesh.T, c2s.T)
        ax[2,i].pcolormesh(x_mesh, y_mesh, c2s, vmin=0, vmax=30, cmap='Greens', rasterized=True) 
        ax[2,i].plot(Pop_living[:,0], Pop_living[:,1],'.', color='k')

        for ii in range(3):
            ax[ii, i].set_xlim(xlim)
            ax[ii, i].set_ylim(ylim)
            ax[ii, i].set_aspect('equal')
            ax[ii, i].set_xticks([])
            ax[ii, i].set_yticks([])

    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    #plt.savefig('chopped.svg',transparent=True)

    cbfig, b = plt.subplots(3,1,figsize=(3,7))
    norm = mcolors.Normalize(vmin=0, vmax=30)
    cbar = mcolorbar.ColorbarBase(b[0], cmap='viridis', norm=norm, orientation='vertical')
    
    norm = mcolors.Normalize(vmin=0, vmax=12)
    cbar = mcolorbar.ColorbarBase(b[1], cmap='Blues', norm=norm, orientation='vertical')

    norm = mcolors.Normalize(vmin=0, vmax=30)
    cbar = mcolorbar.ColorbarBase(b[2], cmap='Greens', norm=norm, orientation='vertical')

    plt.subplots_adjust(left=0.1,right=0.2)
    return


def find_cs_vectorized(x, y, Pop_living):
    # Ensure x and y are arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Extract positions of living individuals
    #Pop_living = Pop[living_indices, :]  # Shape: (num_living, 2)
    
    # Compute c1_tilde for all x, y pairs with Pop_living
    def c1_tilde(x_diff, y_diff):
        return np.exp(-0.5 * (x_diff**2 + y_diff**2) / 0.07**2)
    
    # Compute c2_tilde for all x, y pairs with Pop_living
    def c2_tilde(x_diff, y_diff):
        return np.exp(-0.5 * (x_diff**2 + y_diff**2) / 0.3**2)

    # Add a new axis to x and y for broadcasting with the Pop_living dimension
    x_diff = x[..., np.newaxis] - Pop_living[:, 0]  # Shape: (..., num_living)
    y_diff = y[..., np.newaxis] - Pop_living[:, 1]  # Shape: (..., num_living)

    # Compute the tilde values and sum across the living dimension (axis=-1)
    c1 = np.sum(c1_tilde(x_diff, y_diff), axis=-1)
    c2 = np.sum(c2_tilde(x_diff, y_diff), axis=-1)

    return c1, c2

file = np.load('figure_4C_data.npz')
ts = file['t_record']
Pops = file['Pop_record']
Pop_after_cut = file['Pop_after_cut']

cmap = plt.colormaps['viridis']

print(len(ts))

figure_with_cs([0, 15, 30, 40, 41, 60])
plt.show()