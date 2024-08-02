# Solving a Second Order ODE Boundary Value Problem with Hard Constraints

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

Next, we declare our equation, order, inital values, t boundary, number of points, and epochs. Equation must be in form eqn = 0

    eqn = "utt + 4*u"
    order = 2
    inits = [-2, 10]
    t_bdry = [0,np.pi/4]
    N_pde = 100
    epochs = 1000

To solve, we simply call the corresponding solving function to our problem, and we will declare a hard constraint on boundries

    mymodel = ode_Solvers.solveODE_BVP(eqn, order, inits, t_bdry, N_pde, epochs, constraint = "hard")

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
    epochs = 1000

    mymodel = ode_Solvers.solveODE_BVP(eqn, order, inits, t_bdry, N_pde, epochs)

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()