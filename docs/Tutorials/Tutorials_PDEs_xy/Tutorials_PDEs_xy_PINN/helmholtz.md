# Solving the Helmholtz equation with Hard constrainted Dirichlet boundaries

## Problem
We will look at solving a specific instance of the Helmholtz equation

$$\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + 2\pi^2\sin(2\pi x)\sin(2\pi y)
= -2\pi^2 $$

Over $x\in[0,1], y\in[0,1]$, with boundary conditions

$$u(x, 0) = u(x, 1) = u(0, y) = u(1, y) = 0 $$

## Implementation

First import package. We will only import pde_Solvers and pde_Boundaries_2var from PinnDE as that is all that is needed. Since we need to represent pi for the equation, we also import numpy. We also use tensorflow's sin (we could not use numpy in this instance as with hard constraints we need tensorflows).

    import pinnde.pde_Solvers as pde_Solvers
    import pinnde.pde_Boundaries_2var as pde_Boundaries_2var
    import numpy as np
    import tensorflow as tf

We then setup the boundaries. We just declare the x boundary, y boundary, number of points along bondary, and declare the 
boundary conditions as a python lambda function which **must** be of 2 variable in a time independent equation. Note
how we must use the vairbale declared even just for making the boundaries a constant

    x_bdry = [0, 1]
    y_bdry = [0, 1]
    N_bc = 100
    boundary  = lambda x, y: 0+0*x 
    boundaries = pde_Boundaries_2var.setup_boundaries_dirichlet_xy(x_bdry, y_bdry, N_bc, 
                                                        all_boundaries_cond=boundary)

Next, we declare our equation, number of points, and epochs. Equation must be in form eqn = 0

    eqn = "uxx + uyy + ((np.pi*2)**2)*u + ((np.pi*2)**2)*tf.sin((np.pi*2)*x)*tf.sin((np.pi*2)*y)"
    N_pde = 10000
    epochs = 1500

To solve, we simply call the corresponding solving function to our problem, declaring a hard constraint, and train the model

    mymodel = pde_Solvers.solvePDE_xy(eqn, boundaries, N_pde, constraint = "hard")
    mymodel.train_model(epochs)

If we want to quickly vizualize our data from training we can add after the solving function

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()

## All Code

    import pinnde.pde_Solvers as pde_Solvers
    import pinnde.pde_Boundaries_2var as pde_Boundaries_2var
    import numpy as np
    import tensorflow as tf

    x_bdry = [0, 1]
    y_bdry = [0, 1]
    N_bc = 100
    boundary  = lambda x, y: 0+0*x 
    boundaries = pde_Boundaries_2var.setup_boundaries_dirichlet_xy(x_bdry, y_bdry, N_bc, 
                                                        all_boundaries_cond=boundary)

    eqn = "uxx + uyy + ((np.pi*2)**2)*u + ((np.pi*2)**2)*tf.sin((np.pi*2)*x)*tf.sin((np.pi*2)*y)"
    N_pde = 10000
    epochs = 1500

    mymodel = pde_Solvers.solvePDE_xy(eqn, boundaries, N_pde, constraint = "hard")
    mymodel.train_model(epochs)

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()