# Solving a Third Order ODE Boundary Value Problem with Soft Constraints

## Problem
We will look at solving the ODE

$$u'''(t) - 2u''(t) + u(t) = 0$$

Over $t\in[0,1]$, with boundary values

$$u(0) = 0, u(1) = 0.93, u'(0) = 1, u'(1) = 0.7$$

## Implementation

First import package. We will only import ode_Solvers as that is all that is needed.

    import pinnde.ode_Solvers as ode_Solvers

Next, we declare our equation, order, inital values, t boundary, number of points, and epochs

    eqn = "uttt - 2*utt + u"
    order = 3
    inits = [0, 0.93, 1, 0.7]
    t_bdry = [0,1]
    N_pde = 200
    epochs = 1500

If we also want to change the default number of internal layers (4) and nodes (40) per layer in our PINN, we can declare them as well

    layers = 6
    nodes = 50

To solve, we simply call the corresponding solving function to our problem, and we will not delcare a constraint as it
defaults to soft

    mymodel = ode_Solvers.solveODE_BVP(eqn, order, inits, t_bdry, N_pde, epochs, 
                                            net_layers=layers, net_units=nodes)

If we want to quickly vizualize our data from training we can add after the solving function

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()

## All Code
        
    import pinnde.ode_Solvers as ode_Solvers

    eqn = "uttt - 2*utt + u"
    order = 3
    inits = [0, 0.93, 1, 0.7]
    t_bdry = [0,1]
    N_pde = 200
    epochs = 1500
    layers = 6
    nodes = 50

    mymodel = ode_Solvers.solveODE_BVP(eqn, order, inits, t_bdry, N_pde, epochs, 
                                            net_layers=layers, net_units=nodes)

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()