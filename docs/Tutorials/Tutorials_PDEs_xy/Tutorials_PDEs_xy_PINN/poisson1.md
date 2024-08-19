# Solving the Poisson equation (#1) with Dirichlet boundaries, specifying the PINN architecture

## Problem
We will look at solving a specific instance of the Poisson equation

$$\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = -2\pi^2\cos(\pi x)\sin(\pi y)$$

Over $x\in[-1,1], y\in[-1,1]$, with boundary conditions

$$u(x, -1) = u(x, 1) = u(-1, y) = u(1, y) = \cos(\pi x)\sin(\pi y)$$

## Implementation

First import package. We will only import pde_Solvers and pde_Boundaries_2var from PinnDE as that is all that is needed. Since we need to represent pi for the equation, we also import numpy. We also use tensorflow's sin and cos (we could not use numpy in this instance as with hard constraints we need tensorflows).

    import pinnde.pde_Solvers as pde_Solvers
    import pinnde.pde_Boundaries_2var as pde_Boundaries_2var
    import numpy as np
    import tensorflow as tf

We then setup the boundaries. We just declare the x boundary, y boundary, number of points along bondary, and declare the 
boundary conditions as a python lambda function which **must** be of 2 variable in a time independent equation

    x_bdry = [-1, 1]
    y_bdry = [-1, 1]
    N_bc = 100
    all_boundary = lambda x, y: tf.cos(np.pi*x)*tf.sin(np.pi*y)
    boundaries = pde_Boundaries_2var.setup_boundaries_dirichlet_xy(x_bdry, y_bdry, N_bc, 
                                                        all_boundaries_cond=all_boundary)

Next, we declare our equation, number of points, and epochs. Equation must be in form eqn = 0

    eqn = "uxx + uyy - (-2*np.pi**2*tf.cos(np.pi*x)*tf.sin(np.pi*y))"
    N_pde = 10000
    epochs = 1000

If we also want to change the default number of internal layers (4) and nodes (60) per layer in our PINN, we can declare them as well

    layers = 6
    nodes = 50

To solve, we simply call the corresponding solving function to our problem, and train the model

    mymodel = pde_Solvers.solvePDE_xy(eqn, boundaries, N_pde,
                                    net_layers = layers, net_units = nodes)
    mymodel.train_model(epochs)

If we want to quickly vizualize our data from training we can add after the solving function

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()

## All Code

    import pinnde.pde_Solvers as pde_Solvers
    import pinnde.pde_Boundaries_2var as pde_Boundaries_2var
    import numpy as np
    import tensorflow as tf

    x_bdry = [-1, 1]
    y_bdry = [-1, 1]
    N_bc = 100
    all_boundary = lambda x, y: tf.cos(np.pi*x)*tf.sin(np.pi*y)
    boundaries = pde_Boundaries_2var.setup_boundaries_dirichlet_xy(x_bdry, y_bdry, N_bc, 
                                                        all_boundaries_cond=all_boundary)

    eqn = "uxx + uyy - (-2*np.pi**2*tf.cos(np.pi*x)*tf.sin(np.pi*y))"
    N_pde = 10000
    epochs = 1000
    layers = 6
    nodes = 50

    mymodel = pde_Solvers.solvePDE_xy(eqn, boundaries, N_pde,
                                    net_layers = layers, net_units = nodes)
    mymodel.train_model(epochs)

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()