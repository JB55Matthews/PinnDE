# Solving a Second Order ODE Boundary Value Problem with Hard Constraints with a DeepONet

## Problem
We will look at solving the ODE

$$u''(t) = -4u(t)$$

Over $t\in[0,\pi/4]$, with boundary values

$$u(0) = -2, u(\pi/4) = 10$$

## Implementation

First import package. We will only import ode_Solvers from PinnDE as that is all that is needed. Since we need to represent
pi for the boundary, we also import numpy.

    import pinnde.ode_Solvers as ode_Solvers
    import numpy as np

Next, we declare our equation, order, inital values, t boundary, number of points, sensor_range, num_sensors, and epochs. 
Equation must be in form eqn = 0

    eqn = "utt + 4*u"
    order = 2
    inits = [-2, 10]
    t_bdry = [0,np.pi/4]
    N_pde = 100
    sensor_range = [-2, 2]
    num_sensors = 3000
    epochs = 1500

To solve, we simply call the corresponding solving function to our problem, and we will declare a hard constraint on boundaries

    mymodel = ode_Solvers.solveODE_DeepONet_BVP(eqn, order, inits, t_bdry, N_pde, sensor_range, 
                                                num_sensors, epochs, constraint = "hard")

If we want to quickly vizualize our data from training we can add after the solving function

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()

## All Code

    import pinnde.ode_Solvers as ode_Solvers
    import numpy as np

    eqn = "utt + 4*u"
    order = 2
    inits = [-2, 10]
    t_bdry = [0,np.pi/4]
    N_pde = 100
    sensor_range = [-2, 2]
    num_sensors = 3000
    epochs = 1500

    mymodel = ode_Solvers.solveODE_DeepONet_BVP(eqn, order, inits, t_bdry, N_pde, sensor_range, 
                                                num_sensors, epochs, constraint = "hard")

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()