# Solving the Korteweg–De Vries equation with periodic boundries and Hard constraint Initial condition with a DeepONet

## Problem
We will look at solving a specific instance of the KdV equation

$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} - (-0.0025)\frac{\partial^3 u}{\partial x^3} = 0 $$

Over $t\in[0,1], x\in[-1,1]$, with initial condition

$$u(x, 0) = \cos(\pi x)$$

## Implementation

First import package. We will only import pde_Solvers and pde_Boundries_2var from pinnDE as that is all that is needed. Since we need to represent pi for the initial condition, we also import numpy. We also use tensorflow's cos (we could not use numpy in this instance as with hard constraints we need tensorflows).

    import pinnDE.pde_Solvers as pde_Solvers
    import pinnDE.pde_Boundries_2var as pde_Boundries_2var
    import numpy as np
    import tensorflow as tf

We first can create our initial condition as a python lambda function. This **must** be of 1 variable in a time dependent equation

    u0 = lambda x: tf.cos(np.pi*x)

We then setup the boundries. As we are using periodic boundries this is done easily

    boundries = pde_Boundries_2var.setup_boundries_periodic_tx()

Next, we declare our equation, order of t, initial condition, t boundary, x boundary, number of points, 
number of inital value points, number of sensors, sensor range, and epochs. Equation must be in form eqn = 0

    eqn = "ut+u*ux-(-0.0025)*uxxx"
    t_order = 1
    initial_cond = [u0]
    t_bdry = [0,1]
    x_bdry = [-1,1]
    N_pde = 10000
    N_iv = 100
    N_sensors = 10000
    sensor_range = [-2, 2]
    epochs = 2500

To solve, we simply call the corresponding solving function to our problem, specifying the constraint as hard

    mymodel = pde_Solvers.solvePDE_DeepONet_tx(eqn, t_order, initial_cond, boundries, t_bdry, x_bdry, 
                                    N_pde, N_iv, N_sensors, sensor_range, epochs, constraint = "hard")

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

    eqn = "ut+u*ux-(-0.0025)*uxxx"
    t_order = 1
    initial_cond = [u0]
    t_bdry = [0,1]
    x_bdry = [-1,1]
    N_pde = 10000
    N_iv = 100
    N_sensors = 10000
    sensor_range = [-2, 2]
    epochs = 2500

    mymodel = pde_Solvers.solvePDE_DeepONet_tx(eqn, t_order, initial_cond, boundries, t_bdry, x_bdry, 
                                    N_pde, N_iv, N_sensors, sensor_range, epochs, constraint = "hard")
                                    
    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()