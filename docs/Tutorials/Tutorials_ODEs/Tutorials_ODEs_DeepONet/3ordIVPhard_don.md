# Solving a Third Order ODE Initial Value Problem with Hard Constraints with a DeepONet

## Problem
We will look at solving the ODE

$$u'''(t) - 2u''(t) + u(t) = 0$$

Over $t\in[0,1]$, with inital values

$$u(0) = 0, u'(0) = 1, u''(0) = 0$$

## Implementation

First import package. We will only import ode_Solvers as that is all that is needed.

    import pinnde.ode_Solvers as ode_Solvers

Next, we declare our equation, order, inital values, t boundary, number of points, sensor_range, num_sensors, 
and epochs

    eqn = "uttt - 2*utt + u"
    order = 3
    inits = [0, 1, 0]
    t_bdry = [0,1]
    N_pde = 200
    sensor_range = [-3, 3]
    num_sensors = 4000
    epochs = 2000

To solve, we simply call the corresponding solving function to our problem, and we will declare we want
to hard constraint

    mymodel = ode_Solvers.solveODE_DeepONet_IVP(eqn, order, inits, t_bdry, N_pde, sensor_range, 
                                                    num_sensors, epochs, constraint = "hard")

If we want to quickly vizualize our data from training we can add after the solving function

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()

## All Code

    import pinnde.ode_Solvers as ode_Solvers

    eqn = "uttt - 2*utt + u"
    order = 3
    inits = [0, 1, 0]
    t_bdry = [0,1]
    N_pde = 200
    sensor_range = [-3, 3]
    num_sensors = 4000
    epochs = 2000

    mymodel = ode_Solvers.solveODE_DeepONet_IVP(eqn, order, inits, t_bdry, N_pde, sensor_range, 
                                                    num_sensors, epochs, constraint = "hard")

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()