# Solving the Allen-Cahn equation with Dirichlet boundaries

## Problem
We will look at solving a specific instance of the Allen-Cahn equation (d = 0.0001)

$$\frac{\partial u}{\partial t} = 0.0001\frac{\partial^2 u}{\partial x^2} - 5(u^3 - u) $$

Over $t\in[0,0.5], x\in[-1,1]$, with initial condition

$$u(x, 0) = x^2\cos(\pi x)$$

With our dirichlet boundary conditions defined as

$$u(-1, t) = u(1, t) = -1$$

## Implementation

First import package. We will only import pde_Solvers, pde_Initials and pde_Boundaries_2var from PinnDE as that is all that is needed. Since we need to represent pi for the initial condition, we also import numpy. We also use tensorflow's cos (we could use numpy in this instance, however with hard constraints we need tensorflows so we use this always).

    import pinnde.pde_Solvers as pde_Solvers
    import pinnde.pde_Initials as pde_Initials
    import pinnde.pde_Boundaries_2var as pde_Boundaries
    import numpy as np
    import tensorflow as tf

We first can create our initial condition as a python lambda function. This **must** be of 1 variable in a time dependent equation.
We declare t boundary, x_boundary, the order of t, and number of points along initial t.

    u0 = lambda x: x**2*np.cos(np.pi*x)
    t_bdry = [0, 0.5]
    x_bdry = [-1, 1]
    t_order = 1
    N_iv = 100
    initials = pde_Initials.setup_initials_2var(t_bdry, x_bdry, t_order, [u0], N_iv)

We then setup the boundaries. We just declare the number of points along bondary, and declare the boundary conditions as a python lambda function which **must** be of 1 variable in a time dependent equation. Note how we must use the vairbale declared even just for making the boundaries a constant

    N_bc = 100
    boundary  = lambda x: -1+0*x 
    boundaries = pde_Boundaries.setup_boundaries_dirichlet_tx(t_bdry, x_bdry, N_bc, 
                                                        all_boundaries_cond=boundary)


Next, we declare our equation, number of points,  and epochs. 
Equation must be in form eqn = 0

    eqn = "ut-0.0001*uxx+5*u**3-5*u"
    N_pde = 10000
    epochs = 2000

To solve, we simply call the corresponding solving function to our problem, and train the model

    mymodel = pde_Solvers.solvePDE_tx(eqn, initials, boundaries, N_pde)
    mymodel.train_model(epochs)

If we want to quickly vizualize our data from training we can add after the solving function

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()

## All Code

    import pinnde.pde_Solvers as pde_Solvers
    import pinnde.pde_Initials as pde_Initials
    import pinnde.pde_Boundaries_2var as pde_Boundaries
    import numpy as np
    import tensorflow as tf

    u0 = lambda x: x**2*np.cos(np.pi*x)
    t_bdry = [0, 0.5]
    x_bdry = [-1, 1]
    t_order = 1
    N_iv = 100

    initials = pde_Initials.setup_initials_2var(t_bdry, x_bdry, t_order, [u0], 100)

    N_bc = 100
    boundary  = lambda x: -1+0*x 
    boundaries = pde_Boundaries.setup_boundaries_dirichlet_tx(t_bdry, x_bdry, N_bc, 
                                                        all_boundaries_cond=boundary)

    eqn = "ut-0.0001*uxx+5*u**3-5*u"
    N_pde = 10000
    epochs = 2000

    mymodel = pde_Solvers.solvePDE_tx(eqn, initials, boundaries, N_pde)
    mymodel.train_model(epochs)

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()