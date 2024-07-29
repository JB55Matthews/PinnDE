# Solving the Allen-Cahn equation with Dirichlet boundries

## Problem
We will look at solving a specific instance of the Allen-Cahn equation (d = 0.0001)

$$\frac{\partial u}{\partial t} = 0.0001\frac{\partial^2 u}{\partial x^2} - 5(u^3 - u) $$

Over $t\in[0,0.5], x\in[-1,1]$, with initial condition

$$u(x, 0) = x^2\cos(\pi x)$$

With our dirichlet boundary conditions defined as

$$u(-1, t) = u(1, t) = -1$$

## Implementation

First import package. We will only import pde_Solvers and pde_Boundries_2var from pinnDE as that is all that is needed. Since we need to represent pi for the initial condition, we also import numpy. We also use tensorflow's cos (we could use numpy in this instance, however with hard constraints we need tensorflows so we use this always).

    import pinnDE.pde_Solvers as pde_Solvers
    import pinnDE.pde_Boundries_2var as pde_Boundries_2var
    import numpy as np
    import tensorflow as tf

We first can create our initial condition as a python lambda function. This **must** be of 1 variable in a time dependent equation

    u0 = lambda x: x**2*np.cos(np.pi*x)

We then setup the boundries. We just declare the t boundary, x boundary, number of points along bondary, and declare the 
boundary conditions as a python lambda function which **must** be of 1 variable in a time dependent equation. Note
how we must use the vairbale declared even just for making the boundries a constant

    t_bdry = [0, 0.5]
    x_bdry = [-1, 1]
    N_bc = 100
    boundry  = lambda x: -1+0*x 
    boundries = pde_Boundries.setup_boundries_dirichlet_tx(t_bdry, x_bdry, N_bc, 
                                                        all_boundries_cond=boundry)

Next, we declare our equation, order of t, initial condition, number of points, number of inital value points, and epochs. 
Equation must be in form eqn = 0

    eqn = "ut-0.0001*uxx+5*u**3-5*u"
    t_order = 1
    initial_cond = [u0]
    N_pde = 10000
    N_iv = 200
    epochs = 2000

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

    u0 = lambda x: x**2*np.cos(np.pi*x)

    t_bdry = [0, 0.5]
    x_bdry = [-1, 1]
    N_bc = 100
    boundry  = lambda x: -1+0*x 
    boundries = pde_Boundries.setup_boundries_dirichlet_tx(t_bdry, x_bdry, N_bc, 
                                                        all_boundries_cond=boundry)

    eqn = "ut-0.0001*uxx+5*u**3-5*u"
    t_order = 1
    initial_cond = [u0]
    N_pde = 10000
    N_iv = 200
    epochs = 2000

    mymodel = pde_Solvers.solvePDE_tx(eqn, t_order, initial_cond, boundries, t_bdry, x_bdry, 
                                        N_pde, N_iv, epochs)

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()