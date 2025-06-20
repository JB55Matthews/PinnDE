# Solving a system of odes with a PINN

## Problem
We will look at solving the ODEs

$$u1''(x) + u1(x) = 0$$

$$u2'(x) + u1(x) = 0$$

Over $x\in[0,1]$, with initial values

$$u1(0) = 1/2, u1'(0) = 1$$

$$u2(0) = 2$$

## Implementation
First import package.
    
    import pinnde as p

We then first create our domain. For ODEs, we must use a one dimensional interval, so a NRect with 1 dimension.

    re = p.domain.NRect(1, [0], [1])

We then define the boundaries for the domain. We use the odeicbc boundaries when solving ODEs. As we are doing initial conditions,
we use the flag ic and pass in the initial conditions for each equation

    u1inits = [0.5, 1]
    u2inits = [2]
    cond = p.boundaries.odeicbc(re, [u1inits, u2inits], "ic")

We then create the data for the pinn to train on. As we have no time component and we will use a pinn, we create a pinndata object.

    dat = p.data.pinndata(re, cond, 1000, 100)

Then, we create the pinn model class and train the model for our desired epochs. When defining our equation, all spatial variables are denoted
x1, x2, etc. So our equation is defined as follows.

    eqn1 = "u1x1x1 + u1"
    eqn2 = "u2x1+u1"
    mymodel = p.models.pinn(dat, [eqn1, eqn2])
    mymodel.train(500)

If we want to quickly visualize our solution and epoch loss, we call the in-built plotting functions for this type of equation.

    p.plotters.plot_solution_prediction_1D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)

## All Code

    import pinnde as p

    re = p.domain.NRect(1, [0], [1])

    u1inits = [0.5, 1]
    u2inits = [2]
    cond = p.boundaries.odeicbc(re, [u1inits, u2inits], "ic")

    dat = p.data.pinndata(re, cond, 1000, 100)

    eqn1 = "u1x1x1 + u1"
    eqn2 = "u2x1+u1"
    mymodel = p.models.pinn(dat, [eqn1, eqn2])
    mymodel.train(500)

    p.plotters.plot_solution_prediction_1D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)

Or more concisely,

    import pinnde as p

    re = p.domain.NRect(1, [0], [1])
    cond = p.boundaries.odeicbc(re, [[0.5, 1], [2]], "ic")
    dat = p.data.pinndata(re, cond, 1000, 100)
    mymodel = p.models.pinn(dat, ["u1x1x1 + u1", "u2x1+u1"])
    mymodel.train(500)
    p.plotters.plot_solution_prediction_1D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)