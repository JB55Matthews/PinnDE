# Solving a System of 2 ODEs as Initial Value Problem with soft Constraints

## Problem
We will look at solving the ODEs

$$u''(t) - u'(t)^2 = 0$$

$$x'(t) + u'(t) = x(t)$$

Over $t\in[0,1]$, with initial values

$$u(0) = 1, u'(0) = 1/2$$

$$x(0) = 1$$

## Implementation

First import package. We will only import ode_Solvers from PinnDE as that is all that is needed

    import pinnde.ode_Solvers as ode_Solvers

Next, we declare our equations, orders, inital values, t boundary, number of points, and epochs. Equation must be in form eqn = 0

    eqn1 = "utt - ut**2"
    eqn2 = "xt + ut - x"
    eqns = [eqn1, eqn2]
    orders = [2, 1]
    inits = [[1, 0.5], [1]]
    t_bdry = [0,1]
    N_pde = 100
    epochs = 1000

To solve, we simply call the corresponding solving function to our problem

    mymodel = ode_Solvers.solveODE_System_IVP(eqns, orders, inits, t_bdry, N_pde, epochs)

If we want to quickly vizualize our data from training we can add after the solving function

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()

## All Code

    import pinnde.ode_Solvers as ode_Solvers

    eqn1 = "utt - ut**2"
    eqn2 = "xt + ut - x"
    eqns = [eqn1, eqn2]
    orders = [2, 1]
    inits = [[1, 0.5], [1]]
    t_bdry = [0,1]
    N_pde = 100
    epochs = 1000

    mymodel = ode_Solvers.solveODE_System_IVP(eqns, orders, inits, t_bdry, N_pde, epochs)

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()