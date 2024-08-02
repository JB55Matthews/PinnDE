# Solving a Second Order ODE Initial Value Problem with Soft Constraints

## Problem
We will look at solving the ODE

$$u''(t) = u'(t)^2$$

Over $t\in[0,1]$, with inital values

$$u(0) = 1, u'(0) = 1/2$$

## Implementation

First import package. We will only import ode_Solvers from PinnDE as that is all that is needed. We will also import numpy to plot
the exact solution

    import pinnde.ode_Solvers as ode_Solvers
    import numpy as np

Next, we declare our equation, order, inital values, t boundary, number of points, and epochs. Equation must be in form eqn = 0

    eqn = "utt - ut**2"
    order = 1
    inits = [1, 0.5]
    t_bdry = [0,1]
    N_pde = 100
    epochs = 1000

To solve, we simply call the corresponding solving function to our problem, and we will leave constraint undeclared as it defaults to soft

    mymodel = ode_Solvers.solveODE_IVP(eqn, order, inits, t_bdry, N_pde, epochs)

If we want to quickly vizualize our data from training we can add after the solving function

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()

We can also plot the predicted solution against an exact solution if we have one with

    exact_eqn = "-(np.log(abs(t-2))) + np.log(2) + 1"
    mymodel.plot_predicted_exact(exact_eqn)

## All Code

    import pinnde.ode_Solvers as ode_Solvers
    import numpy as np

    eqn = "utt - ut**2"
    order = 1
    inits = [1, 0.5]
    t_bdry = [0,1]
    N_pde = 100
    epochs = 1000

    mymodel = ode_Solvers.solveODE_IVP(eqn, order, inits, t_bdry, N_pde, epochs)

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()

    exact_eqn = "-(np.log(abs(t-2))) + np.log(2) + 1"
    mymodel.plot_predicted_exact(exact_eqn)