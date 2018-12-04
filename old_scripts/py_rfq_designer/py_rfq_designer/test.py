import bempp.api
import numpy as np

bempp.api.global_parameters.quadrature.medium.double_order = 4
bempp.api.global_parameters.quadrature.far.double_order = 4

grid = bempp.api.import_grid("rfq.msh")

dirichlet_segments = [1, 2]
neumann_segments = [99]

order_neumann = 1
order_dirichlet = 2

global_neumann_space = bempp.api.function_space(grid, "DP", order_neumann)
global_dirichlet_space = bempp.api.function_space(grid, "P", order_dirichlet)

neumann_space_dirichlet_segment = bempp.api.function_space(
    grid, "DP", order_neumann, domains=dirichlet_segments,closed=True, element_on_segment=True)

neumann_space_neumann_segment = bempp.api.function_space(
    grid, "DP", order_neumann, domains=neumann_segments,
    closed=False, element_on_segment=True, reference_point_on_segment=False)

dirichlet_space_dirichlet_segment = bempp.api.function_space(
    grid, "P", order_dirichlet, domains=dirichlet_segments, closed=True)

dirichlet_space_neumann_segment = bempp.api.function_space(
    grid, "P", order_dirichlet, domains=neumann_segments, closed=False)

dual_dirichlet_space = bempp.api.function_space(
    grid, "P", order_dirichlet, domains=dirichlet_segments,
    closed=True, strictly_on_segment=True)

print("Spaces created")

slp_DD = bempp.api.operators.boundary.laplace.single_layer(
    neumann_space_dirichlet_segment,
    dirichlet_space_dirichlet_segment,
    neumann_space_dirichlet_segment)

dlp_DN = bempp.api.operators.boundary.laplace.double_layer(
    dirichlet_space_neumann_segment,
    dirichlet_space_dirichlet_segment,
    neumann_space_dirichlet_segment)

adlp_ND = bempp.api.operators.boundary.laplace.adjoint_double_layer(
    neumann_space_dirichlet_segment,
    neumann_space_neumann_segment,
    dirichlet_space_neumann_segment)

hyp_NN = bempp.api.operators.boundary.laplace.hypersingular(
    dirichlet_space_neumann_segment,
    neumann_space_neumann_segment,
    dirichlet_space_neumann_segment)

slp_DN = bempp.api.operators.boundary.laplace.single_layer(
    neumann_space_neumann_segment,
    dirichlet_space_dirichlet_segment,
    neumann_space_dirichlet_segment)

dlp_DD = bempp.api.operators.boundary.laplace.double_layer(
    dirichlet_space_dirichlet_segment,
    dirichlet_space_dirichlet_segment,
    neumann_space_dirichlet_segment)

id_DD = bempp.api.operators.boundary.sparse.identity(
    dirichlet_space_dirichlet_segment,
    dirichlet_space_dirichlet_segment,
    neumann_space_dirichlet_segment)

adlp_NN = bempp.api.operators.boundary.laplace.adjoint_double_layer(
    neumann_space_neumann_segment,
    neumann_space_neumann_segment,
    dirichlet_space_neumann_segment)

id_NN = bempp.api.operators.boundary.sparse.identity(
    neumann_space_neumann_segment,
    neumann_space_neumann_segment,
    dirichlet_space_neumann_segment)

hyp_ND = bempp.api.operators.boundary.laplace.hypersingular(
    dirichlet_space_dirichlet_segment,
    neumann_space_neumann_segment,
    dirichlet_space_neumann_segment)

print("Operators created")

blocked = bempp.api.BlockedOperator(2, 2)
blocked[0, 0] = slp_DD
blocked[0, 1] = -dlp_DN
blocked[1, 0] = adlp_ND
blocked[1, 1] = hyp_NN

def dirichlet_data_fun(x):
    return 1


def dirichlet_data(x, n, domain_index, res):
    if domain_index == 1:
        res[0] = 22000
    elif domain_index == 2:
        res[0] = -22000
    else:
        res[0] = 0
        
def neumann_data_fun(x):
    return 1

def neumann_data(x, n, domain_index, res):
    res[0] = neumann_data_fun(x)

print("Functions written")
    
dirichlet_grid_fun = bempp.api.GridFunction(dirichlet_space_dirichlet_segment,
                                            fun=dirichlet_data, dual_space=dual_dirichlet_space)

neumann_grid_fun = bempp.api.GridFunction(neumann_space_neumann_segment,
                                          fun=neumann_data, dual_space=dirichlet_space_neumann_segment)

rhs_fun1 = (.5 * id_DD + dlp_DD) * dirichlet_grid_fun - slp_DN * neumann_grid_fun
rhs_fun2 = - hyp_ND * dirichlet_grid_fun + (.5 * id_NN - adlp_NN) * neumann_grid_fun

print("All operators assembled, ready for solve")


lhs = blocked.weak_form()
rhs = np.hstack([rhs_fun1.projections(neumann_space_dirichlet_segment),
                 rhs_fun2.projections(dirichlet_space_neumann_segment)])

from scipy.sparse.linalg import gmres
x, info = gmres(lhs, rhs)

print("Solved")

nx0 = neumann_space_dirichlet_segment.global_dof_count

neumann_solution = bempp.api.GridFunction(
        neumann_space_dirichlet_segment, coefficients=x[:nx0])
dirichlet_solution = bempp.api.GridFunction(
        dirichlet_space_neumann_segment, coefficients=x[nx0:])

neumann_imbedding_dirichlet_segment =  bempp.api.operators.boundary.sparse.identity(
    neumann_space_dirichlet_segment,global_neumann_space,global_neumann_space)

neumann_imbedding_neumann_segment = bempp.api.operators.boundary.sparse.identity(
    neumann_space_neumann_segment,global_neumann_space,global_neumann_space)

dirichlet_imbedding_dirichlet_segment = bempp.api.operators.boundary.sparse.identity(
    dirichlet_space_dirichlet_segment,global_dirichlet_space,global_dirichlet_space)

dirichlet_imbedding_neumann_segment = bempp.api.operators.boundary.sparse.identity(
    dirichlet_space_neumann_segment,global_dirichlet_space,global_dirichlet_space)

print("Assembled solutions spaces")

dirichlet = (dirichlet_imbedding_dirichlet_segment * dirichlet_grid_fun + \
             dirichlet_imbedding_neumann_segment * dirichlet_solution)
neumann = (neumann_imbedding_neumann_segment * neumann_grid_fun + \
           neumann_imbedding_dirichlet_segment * neumann_solution)

dirichlet.plot()
