import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from model_utils import Model
from parameters import params

model = Model(*params)


def dYdt(Y, t):
    V1, V2 = model.V(Y[0], Y[1], lambdaD=lambdaD)
    return [V1, V2]


def plot_curve_with_arrow(ax, xs, ys, length_scale=5, angle=0.5, 
                          dv_adjust_angle=0, xy_scale_ratio=None,
                          linewidth=1, color='k'):
    # xy_scale_ratio = x_scale = ((xmax-xmin)/width)  /  ((ymax-ymin)/height)

    def R(theta):  # Rotation matrix
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    if xy_scale_ratio is None:  # Calculate xy_scale_ratio from ax
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        fig = ax.get_figure()
        bbox = ax.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        xy_scale_ratio = ((xmax-xmin)/width) / ((ymax-ymin)/height)

    dx = xs[-1] - xs[-2]
    dy = ys[-1] - ys[-2]
    dv = R(dv_adjust_angle)@[dx, dy]
    T = np.array([[1/xy_scale_ratio, 0], [0, 1]])
    T_inv = np.array([[xy_scale_ratio, 0], [0, 1]])

    dv_prime = T@dv
    s1_prime = R(angle)@dv_prime
    s2_prime = R(-angle)@dv_prime

    s1 = length_scale*T_inv@s1_prime
    s2 = length_scale*T_inv@s2_prime

    xs = np.append(
        xs, [xs[-1] - s1[0], xs[-1], xs[-1] - s2[0], xs[-1] - 0.5*s2[0]])
    ys = np.append(
        ys, [ys[-1] - s1[1], ys[-1], ys[-1] - s2[1], ys[-1] - 0.5*s2[1]])

    plt.plot(xs, ys, linewidth=linewidth, color=color)


S_th = 1.5  # xB threshold
D_th = 0.5  # xB threshold

if 1:  # Plot phase portraits
    y1_range = np.linspace(-.1, 4.1, 50)
    y2_range = np.linspace(-.1, 2.6, 50)
    y1_mesh, y2_mesh = np.meshgrid(y1_range, y2_range)

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(9, 2.7))

    stream_density = 0.7

    # a
    lambdaD = 0
    V1, V2 = model.V(y1_mesh, y2_mesh, lambdaD=lambdaD)
    a.streamplot(y1_mesh, y2_mesh, V1, V2, density=stream_density)

    Y1s = [[1.5, 2], [1.5, .1]]
    for Y1 in Y1s:
        Y1_of_t = odeint(dYdt, Y1, np.linspace(0, 20, 10))
        a.plot(Y1_of_t[-1, 0], Y1_of_t[-1, 1], 's', color='k', markersize=10)

    # b
    lambdaD = 30
    V1, V2 = model.V(y1_mesh, y2_mesh, lambdaD=lambdaD)
    b.streamplot(y1_mesh, y2_mesh, V1, V2, density=stream_density)

    Y1s = [[.1, 2.], [1.5, 2], [1.5, .1]]
    for Y1 in Y1s:
        Y1_of_t = odeint(dYdt, Y1, np.linspace(0, 20, 10))
        b.plot(Y1_of_t[-1, 0], Y1_of_t[-1, 1], 's', color='k', markersize=10)

    b.set_yticklabels([''])

    # c
    lambdaD = 50
    V1, V2 = model.V(y1_mesh, y2_mesh, lambdaD=lambdaD)
    c.streamplot(y1_mesh, y2_mesh, V1, V2, density=stream_density)

    Y1s = [[.1, 2.], [1.5, 2], [1.5, .1]]
    for Y1 in Y1s:
        Y1_of_t = odeint(dYdt, Y1, np.linspace(0, 20, 10))
        c.plot(Y1_of_t[-1, 0], Y1_of_t[-1, 1], 's', color='k', markersize=10)

    c.set_yticklabels([''])

    for ax in [a, b, c]:
        ax.plot([0, 4], [S_th, S_th], '--', color='k', linewidth=1.8)
        ax.plot([0, 4], [D_th, D_th], '--', color='k', linewidth=1.8)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 2.5)

    plt.subplots_adjust(wspace=.1)

    plt.savefig('phase_portraits.pdf', transparent=True)
    plt.show()

if 1:  # Plot equilibrium manifolds
    ax = plt.figure().add_subplot(projection='3d')
    lambdaDs = np.linspace(0, 50, 200)
    for lambdaD in lambdaDs:
        Y1s = [[-1, 0], [2, 0], [3, 3]]
        for Y1 in Y1s:
            Y1_of_t = odeint(dYdt, Y1, np.linspace(0, 100, 10))
            ax.plot(Y1_of_t[-1, 0], Y1_of_t[-1, 1], lambdaD, 'o', color='k')
    # ax.set_xlabel('$y_1$')
    # ax.set_ylabel('$y_2$')
    # ax.set_zlabel(r'$\lambda_2$')
    plt.show()
