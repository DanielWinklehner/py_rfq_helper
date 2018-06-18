import bempp.api
import numpy as np
from scipy import meshgrid
from matplotlib import pyplot as plt

grid = bempp.api.import_grid('TransitionCell_Assy.msh')

# grid1 = bempp.api.shapes.sphere(origin=(-2.0, 0.0, 0.0), h=0.5)
# grid2 = bempp.api.shapes.sphere(origin=(2.0, 0.0, 0.0), h=0.5)
#
# no_vert_grid1 = grid1.leaf_view.vertices.shape[1]
#
# vertices = np.append(grid1.leaf_view.vertices, grid2.leaf_view.vertices, axis=1)
# elements = np.append(grid1.leaf_view.elements, grid2.leaf_view.elements+no_vert_grid1, axis=1)
#
# grid = bempp.api.grid.grid_from_element_data(vertices, elements)

grid.plot()

space = bempp.api.function_space(grid, "DP", 0)
slp = bempp.api.operators.boundary.laplace.single_layer(space, space, space)


def f(r, n, domain_index, result):
    if abs(r[0]) > 11.0:
        result[0] = 25000.0
    else:
        result[0] = -25000.0
    # result[0] = x[0] + 1

rhs = bempp.api.GridFunction(space, fun=f)

sol, _ = bempp.api.linalg.gmres(slp, rhs)

sol.plot()

# xy_lim = 4.0
# res = 0.5
# xy_lim = 10.0
# z_lim = 250.0
#
# nx = ny = np.round(2.0 * xy_lim / res, 0) + 1
# nz = np.round(z_lim / res, 0) + 1
#
# x = np.linspace(-xy_lim, xy_lim, nx)
# y = np.linspace(-xy_lim, xy_lim, ny)
# z = np.linspace(0.0, z_lim, nz)
#
# grid_x, grid_y, grid_z = meshgrid(x, y, z)
#
# points = np.vstack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()])
#
# nearfield = bempp.api.operators.potential.laplace.single_layer(space, points)
# pot_discreet = nearfield * sol
#
# idx = np.where((points[0] == 0) & (points[1] == 0))
# ez = -np.gradient(pot_discreet[0][idx], res)
#
# plt.plot(z, ez)
# plt.show()

nvals = 251
z_vals = np.linspace(0.0, 250.0, nvals)
points = np.vstack([np.zeros(nvals), np.zeros(nvals), z_vals])
nearfield = bempp.api.operators.potential.laplace.single_layer(space, points)
pot_discrete = nearfield * sol

plt.plot(z_vals, pot_discrete[0])
plt.show()

ez = -np.gradient(pot_discrete[0], 250.0/nvals)
plt.plot(z_vals, ez)
plt.show()
