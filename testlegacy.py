# import pinnde.pde_Solvers as pde_Solvers
# import pinnde.pde_Initials as pde_Initials
# import pinnde.pde_Boundaries_2var as pde_Boundaries_2var

# import src.pinnde.legacy.pde_Solvers as pde_Solvers
# import src.pinnde.legacy.pde_Initials as pde_Initials
# import src.pinnde.legacy.pde_Boundaries_2var as pde_Boundaries_2var


import numpy as np
import tensorflow as tf

# u0 = lambda x: tf.cos(np.pi*x)
# t_bdry = [0,1]
# x_bdry = [-1,1]
# t_order = 1
# N_iv = 100
# initials = pde_Initials.setup_initials_2var(t_bdry, x_bdry, t_order, [u0], N_iv)

# boundaries = pde_Boundaries_2var.setup_boundaries_periodic_tx(t_bdry, x_bdry)

# eqn = "ut+ux"
# N_pde = 10000
# epochs = 300

# mymodel = pde_Solvers.solvePDE_tx(eqn, initials, boundaries, N_pde)
# mymodel.train_model(epochs)

# mymodel.plot_epoch_loss()

# mymodel.plot_solution_prediction()

# import src.pinnde.legacy.PDE.pde_Points as p
# pdes, usensors = p.defineCollocationPoints_DON_2var([0, 1], [0,1], 100, 10, 200, [-2,2])