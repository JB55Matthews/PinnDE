# Solving an ode with a DeepONet

## Problem
We will look at solving the ODE

$$u''(x) + u(x) = 0$$

Over $x\in[0,1]$, with initial value

$$u(0) = 1/2, u'(0) = 1$$

## Implementation
First import package.
    
    import pinnde as p

We then first create our domain. For ODEs, we must use a one dimensional interval, so a NRect with 1 dimension.

    re = p.domain.NRect(1, [0], [1])

We then define the boundaries for the domain. We use the odeicbc boundaries when solving ODEs. As we are doing initial conditions,
we use the flag ic and pass in the initial conditions.

    uinits = [0.5, 1]
    cond = p.boundaries.odeicbc(re, [uinits], "ic")

We then create the data for the deeponet to train on. As we have no time component and we will use a deeponet, we create a dondata object.

    dat = p.data.dondata(re, cond, 1000, 100, 500)

Then, we create the deeponet model class and train the model for our desired epochs. When defining our equation, all spatial variables are denoted
x1, x2, etc. So our equation is defined as follows.

    eqn = "ux1x1 + u"
    mymodel = p.models.pinn(dat, [eqn])
    mymodel.train(1000)

If we want to quickly visualize our solution and epoch loss, we call the in-built plotting functions for this type of equation.

    p.plotters.plot_solution_prediction_1D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)

## All Code

    import pinnde as p

    re = p.domain.NRect(1, [0], [1])

    uinits = [0.5, 1]
    cond = p.boundaries.odeicbc(re, [uinits], "ic")

    dat = p.data.dondata(re, cond, 1000, 100, 500)

    eqn = "ux1x1 + u"
    mymodel = p.models.deeponet(dat, [eqn])
    mymodel.train(1000)

    p.plotters.plot_solution_prediction_1D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)

Or more concisely,

    import pinnde as p

    re = p.domain.NRect(1, [0], [1])
    cond = p.boundaries.odeicbc(re, [0.5, 1], "ic")
    dat = p.data.pinndata(re, cond, 1000, 100, 500)
    mymodel = p.models.pinn(dat, ["ux1x1 + u"])
    mymodel.train(1000)
    p.plotters.plot_solution_prediction_1D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)