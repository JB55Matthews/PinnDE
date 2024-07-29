# Solving the Linear Advection equation with Periodic boundries

## Problem
We will look at solving the Linear Advection equation

$$\frac{\partial u}{\partial t} + \frac{\partial u}{\partial x} = 0$$

Over $t\in[0,1], x\in[-1,1]$, with initial condition

$$u(x, 0) = \cos(\pi x)$$

## Implementation

First import package. We will only import pde_Solvers and pde_Boundries_2var from pinnDE as that is all that is needed. Since we need to represent pi for the initial condition, we also import numpy. We also use tensorflow's cos (we could use numpy in this instance, however with hard constraints we need tensorflows so we use this always).

    import pinnDE.pde_Solvers as pde_Solvers
    import pinnDE.pde_Boundries_2var as pde_Boundries_2var
    import numpy as np
    import tensorflow as tf

We first can create our initial condition as a python lambda function. This **must** be of 1 variable in a time dependent equation

    u0 = lambda x: tf.cos(np.pi*x)

We then setup the boundries. As we are using periodic boundries this is done easily

    boundries = pde_Boundries_2var.setup_boundries_periodic_tx()

Next, we declare our equation, order of t, initial condition, t boundary, x boundary, number of points, 
number of inital value points, and epochs. Equation must be in form eqn = 0

    eqn = "ut+ux"
    t_order = 1
    initial_cond = [u0]
    t_bdry = [0,1]
    x_bdry = [-1,1]
    N_pde = 10000
    N_iv = 100
    epochs = 1200

To solve, we simply call the corresponding solving function to our problem

    mymodel = pde_Solvers.solvePDE_tx(eqn, t_order, initial_cond, boundries, t_bdry, x_bdry, 
                                        N_pde, N_iv, epochs)

If we want to quickly vizualize our data from training we can add after the solving function

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()

## All Code

    import pinnDE.pde_Solvers as pde_Solvers
    import pinnDE.pde_Boundries_2var as pde_Boundries_2var
    import numpy as np
    import tensorflow as tf

    u0 = lambda x: tf.cos(np.pi*x)

    boundries = pde_Boundries_2var.setup_boundries_periodic_tx()

    eqn = "ut+ux"
    t_order = 1
    initial_cond = [u0]
    t_bdry = [0,1]
    x_bdry = [-1,1]
    N_pde = 10000
    N_iv = 100
    epochs = 1200

    mymodel = pde_Solvers.solvePDE_tx(eqn, t_order, initial_cond, boundries, t_bdry, x_bdry, 
                                        N_pde, N_iv, epochs)

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()