# Solving the Burgers equation with Periodic boundries

## Problem
We will look at solving a specific instance of the Burgers equation (v = $\frac{0.01}{\pi}$)

$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = (\frac{0.01}{\pi})\frac{\partial^2 u}{\partial x^2} $$

Over $t\in[0,1], x\in[-1,1]$, with initial condition

$$u(x, 0) = -\sin(\pi x)$$

## Implementation

First import package. We will only import pde_Solvers and pde_Boundries_2var from PinnDE as that is all that is needed. Since we need to represent pi for the initial condition, we also import numpy. We also use tensorflow's sin (we could use numpy in this instance, however with hard constraints we need tensorflows so we use this always).

    import pinnde.pde_Solvers as pde_Solvers
    import pinnde.pde_Boundries_2var as pde_Boundries_2var
    import numpy as np
    import tensorflow as tf

We first can create our initial condition as a python lambda function. This **must** be of 1 variable in a time dependent equation

    u0 = lambda x: -tf.sin(np.pi*x)

We then setup the boundries. As we are using periodic boundries this is done easily

    boundries = pde_Boundries_2var.setup_boundries_periodic_tx()

Next, we declare our equation, order of t, initial condition, t boundary, x boundary, number of points, 
number of inital value points, and epochs. Equation must be in form eqn = 0

    eqn = "ut+u*ux-(0.01/np.pi)*uxx"
    t_order = 1
    initial_cond = [u0]
    t_bdry = [0,1]
    x_bdry = [-1,1]
    N_pde = 10000
    N_iv = 100
    epochs = 1000

To solve, we simply call the corresponding solving function to our problem

    mymodel = pde_Solvers.solvePDE_tx(eqn, t_order, initial_cond, boundries, t_bdry, x_bdry, 
                                        N_pde, N_iv, epochs)

If we want to quickly vizualize our data from training we can add after the solving function

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()

## All Code

    import pinnde.pde_Solvers as pde_Solvers
    import pinnde.pde_Boundries_2var as pde_Boundries_2var
    import numpy as np
    import tensorflow as tf

    u0 = lambda x: -tf.sin(np.pi*x)

    boundries = pde_Boundries_2var.setup_boundries_periodic_tx()

    eqn = "ut+u*ux-(0.01/np.pi)*uxx"
    t_order = 1
    initial_cond = [u0]
    t_bdry = [0,1]
    x_bdry = [-1,1]
    N_pde = 10000
    N_iv = 100
    epochs = 1000

    mymodel = pde_Solvers.solvePDE_tx(eqn, t_order, initial_cond, boundries, t_bdry, x_bdry, 
                                        N_pde, N_iv, epochs)

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()