# Solving the Helmholtz equation with Hard constrainted Dirichlet boundries

## Problem
We will look at solving a specific instance of the Helmholtz equation

$$\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + 2\pi^2\sin(2\pi x)\sin(2\pi y)
= -2\pi^2 $$

Over $x\in[0,1], y\in[0,1]$, with boundary conditions

$$u(x, 0) = u(x, 1) = u(0, y) = u(1, y) = 0 $$

## Implementation

First import package. We will only import pde_Solvers and pde_Boundries_2var from pinnDE as that is all that is needed. Since we need to represent pi for the equation, we also import numpy. We also use tensorflow's sin (we could not use numpy in this instance as with hard constraints we need tensorflows).

    import pinnDE.pde_Solvers as pde_Solvers
    import pinnDE.pde_Boundries_2var as pde_Boundries_2var
    import numpy as np
    import tensorflow as tf

We then setup the boundries. We just declare the x boundary, y boundary, number of points along bondary, and declare the 
boundary conditions as a python lambda function which **must** be of 2 variable in a time independent equation. Note
how we must use the vairbale declared even just for making the boundries a constant

    x_bdry = [0, 1]
    y_bdry = [0, 1]
    N_bc = 100
    boundry  = lambda x, y: 0+0*x 
    boundries = pde_Boundries.setup_boundries_dirichlet_xy(x_bdry, y_bdry, N_bc, 
                                                        all_boundries_cond=boundry)

Next, we declare our equation, number of points, number of sensors, sensor range, and epochs. Equation must be in form eqn = 0

    eqn = "uxx + uyy + ((np.pi*2)**2)*u + ((np.pi*2)**2)*tf.sin((np.pi*2)*x)*tf.sin((np.pi*2)*y)"
    N_pde = 10000
    epochs = 1500
    N_sensors = 40000
    sensor_range = [-2, 2]

To solve, we simply call the corresponding solving function to our problem, declaring a hard constraint

    mymodel = pde_Solvers.solvePDE_DeepONet_xy(eqn, boundries, x_bdry, y_bdry, N_pde, N_sensors, 
                                                    sensor_range, epochs, constraint = "hard")

If we want to quickly vizualize our data from training we can add after the solving function

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()

## All Code

    import pinnDE.pde_Solvers as pde_Solvers
    import pinnDE.pde_Boundries_2var as pde_Boundries_2var
    import numpy as np
    import tensorflow as tf

    x_bdry = [0, 1]
    y_bdry = [0, 1]
    N_bc = 100
    boundry  = lambda x, y: 0+0*x 
    boundries = pde_Boundries.setup_boundries_dirichlet_xy(x_bdry, y_bdry, N_bc, 
                                                        all_boundries_cond=boundry)

    eqn = "uxx + uyy + ((np.pi*2)**2)*u + ((np.pi*2)**2)*tf.sin((np.pi*2)*x)*tf.sin((np.pi*2)*y)"
    N_pde = 10000
    epochs = 1500
    N_sensors = 40000
    sensor_range = [-2, 2]

    mymodel = pde_Solvers.solvePDE_DeepONet_xy(eqn, boundries, x_bdry, y_bdry, N_pde, N_sensors, 
                                                    sensor_range, epochs, constraint = "hard")

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()