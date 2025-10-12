import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

def plot_voronoi(axes, points, max_radius, xlim, ylim, cell_colors=None, point_size=3, boundary_width=2):

    # Augment points with 4 corner points to form a very large box around points.
    # This ensures every point in points is enclosed within a cell.
    center_x = (xlim[0] + xlim[1])/2
    center_y = (ylim[0] + ylim[1])/2
    max_displacement = max(np.concatenate((np.abs(points[:,0]-center_x), np.abs(points[:,1]-center_y))))
    aug_points = 100*max_displacement*np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    aug_points[:,0] += center_x
    aug_points[:,1] += center_y
    all_points = np.concatenate((points, aug_points))

    # Scipy voronoi object
    vor = Voronoi(all_points)

    if cell_colors is None:
        cell_colors = ['green']*len(points)

    # for each point, plot the intersection of its voronoi cell and a
    # circle of radius max_radius.
    for i, point in enumerate(points):
        region_index = vor.point_region[i]
        region = vor.regions[region_index]
        verts = vor.vertices[region]
        
        polygon = Polygon(verts)
        circle = Point(point).buffer(max_radius)
        intersection = polygon.intersection(circle)
        xs, ys = intersection.exterior.xy

        axes.plot(point[0],point[1],'o', color='k', markersize=point_size)
        axes.fill(xs, ys, alpha=0.5, fc=cell_colors[i], ec='none')
        axes.plot(xs, ys, color='k', linewidth=boundary_width)

    axes.set_xlim(xlim)
    axes.set_ylim(ylim)

    return