# Solving the Korteweg–De Vries equation with periodic boundaries and Hard constraint Initial condition with a DeepONet

## Problem
We will look at solving a specific instance of the KdV equation

$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} - (-0.0025)\frac{\partial^3 u}{\partial x^3} = 0 $$

Over $t\in[0,1], x\in[-1,1]$, with initial condition

$$u(x, 0) = \cos(\pi x)$$

## Implementation

First import package. We will only import pde_Solvers, pde_Initials, and pde_Boundaries_2var from PinnDE as that is all that is needed. Since we need to represent pi for the initial condition, we also import numpy. We also use tensorflow's cos (we could not use numpy in this instance as with hard constraints we need tensorflows).

    import pinnde.pde_Solvers as pde_Solvers
    import pde_Initials as pde_Initials
    import pinnde.pde_Boundaries_2var as pde_Boundaries_2var
    import numpy as np
    import tensorflow as tf

We first can create our initial condition as a python lambda function. This **must** be of 1 variable in a time dependent equation.
We declare t boundary, x_boundary, the order of t, and number of points along initial t.

    u0 = lambda x: tf.cos(np.pi*x)
    t_bdry = [0,1]
    x_bdry = [-1,1]
    t_oder = 1
    N_iv = 100
    initials = pde_Initials.setup_initials_2var(t_bdry, x_bdry, t_order, [u0], N_iv)

We then setup the boundaries. As we are using periodic boundaries this is done easily

    boundaries = pde_Boundaries_2var.setup_boundaries_periodic_tx(t_bdry, x_bdry)

Next, we declare our equation,  number of points, number of sensors, sensor range, and epochs. Equation must be in form eqn = 0

    eqn = "ut+u*ux-(-0.0025)*uxxx"
    N_pde = 10000
    N_sensors = 10000
    sensor_range = [-2, 2]
    epochs = 2500

To solve, we simply call the corresponding solving function to our problem, specifying the constraint as hard, and train the model

    mymodel = pde_Solvers.solvePDE_DeepONet_tx(eqn, initials, boundaries,  N_pde, 
                                            N_sensors, sensor_range, constraint = "hard")
    mymodel.train_model(epochs)

If we want to quickly vizualize our data from training we can add after the solving function

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()

## All Code

    import pinnde.pde_Solvers as pde_Solvers
    import pinnde.pde_Boundaries_2var as pde_Boundaries_2var
    import numpy as np
    import tensorflow as tf

    u0 = lambda x: tf.cos(np.pi*x)
    t_bdry = [0,1]
    x_bdry = [-1,1]
    t_oder = 1
    N_iv = 100
    initials = pde_Initials.setup_initials_2var(t_bdry, x_bdry, t_order, [u0], N_iv)

    boundaries = pde_Boundaries_2var.setup_boundaries_periodic_tx(t_bdry, x_bdry)

    eqn = "ut+u*ux-(-0.0025)*uxxx"
    N_pde = 10000
    N_sensors = 10000
    sensor_range = [-2, 2]
    epochs = 2500

    mymodel = pde_Solvers.solvePDE_DeepONet_tx(eqn, initials, boundaries,  N_pde, 
                                            N_sensors, sensor_range, constraint = "hard")
    mymodel.train_model(epochs)
    
    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()