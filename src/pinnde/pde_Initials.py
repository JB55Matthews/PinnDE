import numpy as np
from pyDOE import lhs

def setup_initials_2var(t_bdry, x_bdry, t_order, initial_t, N_iv=100):
    """
    Main function for setting up initial conditions for equations in t and x

    Args:
        t_bdry (list): list of two elements, the interval of t to be solved on.
        t_order (int): Order of t in equation (highest derivative of t used). Can be 1-3
        initial_t (list/Lambda): List of Inital functions for t=t0, as a python lambda funcitons, with t0 being inital t in t_bdry.
            One for each order. **Must** be of only 1 variable. See examples for how to make.
        N_iv (int): Number of randomly sampled collocation points along inital t which PINN uses in training.

    Returns:
        inits (list): List of setup sampled initial condition points
        t_order (int): Order of t in equation
        initial_t (list/Lambda): List of Inital functions for t=t0, as a python lambda funcitons, with t0 being inital t in t_bdry.
            One for each order.
        N_iv (int): Number of randomly sampled collocation points along inital t which PINN uses in training.

    """

    if t_order == 1:
        u0 = initial_t[0]
    elif t_order == 2:
        u0, ut0 = initial_t[0], initial_t[1]
    elif t_order == 3:
        u0, ut0 = initial_t[0], initial_t[1]
        utt0 = initial_t[2]

    # Sample points where to evaluate the initial values
    tx_min = np.array([t_bdry[0], x_bdry[0]])
    tx_max = np.array([t_bdry[1], x_bdry[1]])  
    init_points = tx_min[1:] + (tx_max[1:] - tx_min[1:])*lhs(1, N_iv)

    x_init = init_points
    t_init = t_bdry[0]+ 0.0*x_init
    if t_order == 1:
        u_init = u0(x_init)
        inits = np.column_stack([t_init, x_init, u_init]).astype(np.float64)
    elif t_order == 2:
        u_init = u0(x_init)
        ut_init = ut0(x_init)
        inits = np.column_stack([t_init, x_init, u_init, ut_init]).astype(np.float64)
    elif t_order == 3:
        u_init = u0(x_init)
        ut_init = ut0(x_init)
        utt_init = utt0(x_init)
        inits = np.column_stack([t_init, x_init, u_init, ut_init, utt_init]).astype(np.float64)

    return [inits, t_order, initial_t, N_iv]