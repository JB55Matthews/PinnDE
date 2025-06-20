# Solving the 1+1 linear advection equation with periodic boundaries with a DeepONet

## Problem
We will look at solving the linear advection equation

$$\frac{\partial u}{\partial t} + \frac{\partial u}{\partial x} = 0$$

Over $t\in[0,1], x\in[-1,1]$, with initial condition

$$u(x, 0) = \cos(\pi x)$$

## Implementation
First import package. Since we need to represent pi for the initial condition, we also import numpy. We also use tensorflow's cos.
    
    import pinnde as p
    import numpy as np
    import tensorflow as tf

We then first create our domain. We will solve this on the interval -1 to 1, and this equation has a time component, so
we create a Time_NRect with 1 spatial dimensions, with xmins [-1] and xmaxs [1]. We will solve
from time t=0 to t=1.

    timerange = [0, 1]
    tre = p.domain.Time_NRect(1, [-1], [1], timerange)

We then define the boundaries for the domain. We will use periodic boundaries, and so we simply call the function

    bound = p.boundaries.periodic(tre)

As we have a time component, we also must define our initial condition. Derivatives of t has order 1, so we have one initial function.
This is done similarly to our boundaries.

    u0func = lambda x1: tf.cos(np.pi*x1)
    inits = p.initials.initials(tre, [u0func])

We then create the data for the deeponet to train on. As we have a time component and we will use a deeponet, we create a timedondata object.
Note that since we use periodic boundaries, no points will be used in training, however we still sample dummy points, so we provide only 
10 to not slow down training.

    dat = p.data.timedondata(tre, bound, inits, 12000, 10, 600, 600)

Then, we create the deeponet model class and train the model for our desired epochs. When defining our equation, all spatial variables are denoted
x1, x2, etc. So our equation is defined as follows. We also show how to define the internal layers and nodes to use.

    eqn = "ut+ux1"
    mymodel = p.models.deeponet(dat, [eqn], layers=6, units=40)
    mymodel.train(500)

If we want to quickly visualize our solution and epoch loss, we call the in-built plotting functions for this type of equation.

    p.plotters.plot_solution_prediction_time1D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)

## All Code

    import pinnde as p
    import numpy as np
    import tensorflow as tf

    timerange = [0, 1]
    tre = p.domain.Time_NRect(1, [-1], [1], timerange)

    bound = p.boundaries.periodic(tre)

    u0func = lambda x1: tf.cos(np.pi*x1)
    inits = p.initials.initials(tre, [u0func])

    dat = p.data.timedondata(tre, bound, inits, 12000, 10, 600, 600)

    eqn = "ut+ux1"
    mymodel = p.models.deeponet(dat, [eqn], layers=6, units=40)
    mymodel.train(500)

    p.plotters.plot_solution_prediction_time1D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)

Or more concisely,

    import pinnde as p
    import numpy as np
    import tensorflow as tf

    tre = p.domain.Time_NRect(1, [-1], [1], [0,1])
    bound = p.boundaries.periodic(tre)
    inits = p.initials.initials(tre, [lambda x1: tf.cos(np.pi*x1)])
    dat = p.data.timedondata(tre, bound, inits, 10000, 10, 600, 600)
    mymodel = p.models.deeponet(dat, ["ut+ux1"], layers=6, units=40)
    mymodel.train(500)
    p.plotters.plot_solution_prediction_time1D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)