# Solving the Heat equation with Periodic boundaries with a DeepONet

## Problem
We will look at solving a specific instance of the Heat equation

$$ 0.08\frac{\partial^2 u}{\partial x^2}= \frac{\partial u}{\partial t} $$

Over $t\in[0,1], x\in[0,1]$, with initial condition

$$u(x, 0) = \sin(\pi x)$$

## Implementation

First import package. We will only import pde_Solvers, pde_Initials, and pde_Boundaries_2var from PinnDE as that is all that is needed. Since we need to represent pi for the initial condition, we also import numpy. We also use tensorflow's sin (we could use numpy in this instance, however with hard constraints we need tensorflows so we use this always).

    import pinnde.pde_Solvers as pde_Solvers
    import pinnde.pde_Initials as pde_Initials
    import pinnde.pde_Boundaries_2var as pde_Boundaries_2var
    import numpy as np
    import tensorflow as tf

We first can create our initial condition as a python lambda function. This **must** be of 1 variable in a time dependent equation.
We declare t boundary, x_boundary, the order of t, and number of points along initial t.

    u0 = lambda x: tf.sin(np.pi*x)
    t_bdry = [0,1]
    x_bdry = [0,1]
    t_order = 1
    N_iv = 100
    initials = pde_Initials.setup_initials_2var(t_bdry, x_bdry, t_order, [u0], N_iv)

We then setup the boundaries. As we are using periodic boundaries this is done easily

    boundaries = pde_Boundaries_2var.setup_boundaries_periodic_tx(t_bdry, x_bdry)

Next, we declare our equation, number of points, number of sensors, sensor range, and epochs. Equation must be in form eqn = 0

    eqn = "0.08*uxx - ut"
    N_pde = 10000
    N_sensors = 50000
    sensor_range = [-2, 2]
    epochs = 1000

To solve, we simply call the corresponding solving function to our problem, and train the model

    mymodel = pde_Solvers.solvePDE_DeepONet_tx(eqn, initials, boundaries, N_pde,
                                                 N_sensors, sensor_range)
    mymodel.train_model(epochs)

If we want to quickly vizualize our data from training we can add after the solving function

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()

## All Code

    import pinnde.pde_Solvers as pde_Solvers
    import pinnde.pde_Initials as pde_Initials
    import pinnde.pde_Boundaries_2var as pde_Boundaries_2var
    import numpy as np
    import tensorflow as tf

    u0 = lambda x: tf.sin(np.pi*x)
    t_bdry = [0,1]
    x_bdry = [0,1]
    t_order = 1
    N_iv = 100
    initials = pde_Initials.setup_initials_2var(t_bdry, x_bdry, t_order, [u0], N_iv)

    boundaries = pde_Boundaries_2var.setup_boundaries_periodic_tx(t_bdry, x_bdry)

    eqn = "0.08*uxx - ut"
    N_pde = 10000
    N_sensors = 50000
    sensor_range = [-2, 2]
    epochs = 1000

    mymodel = pde_Solvers.solvePDE_DeepONet_tx(eqn, initials, boundaries, N_pde,
                                                 N_sensors, sensor_range)
    mymodel.train_model(epochs)
    
    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()