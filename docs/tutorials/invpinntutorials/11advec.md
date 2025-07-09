# Solving for advcection speed of the 1+1 linear advection equation with periodic boundaries with an Inverse PINN

## Problem
We will look at solving for the constant c in the linear advection equation

$$\frac{\partial u}{\partial t} + c\frac{\partial u}{\partial x} = 0$$

Over $t\in[0,1], x\in[-1,1]$, with initial condition

$$u(x, 0) = \cos(\pi x)$$

## Implementation
First import package. Since we need to represent pi for the initial condition, we also import numpy. We also use tensorflow's cos.
    
    import pinnde as p
    import numpy as np
    import tensorflow as tf

We first then generate our sample data. We have the reference solution to this problem, with c=1, and so we generate data along t, x1, 
and for the reference solution u to mimic having experimental data.

    c = 1
    u0 = lambda x: tf.cos(np.pi*x)
    u_true = lambda x, t: u0(x-c*t)
    N_data = 200
    xdata = np.random.uniform(-1, 1, N_data)
    tdata = np.random.uniform(0, 1, N_data)
    udata = u_true(xdata, tdata)

We then create our domain. We will solve this on the interval -1 to 1, and this equation has a time component, so
we create a Time_NRect with 1 spatial dimensions, with xmins [-1] and xmaxs [1]. We will solve
from time t=0 to t=1.

    timerange = [0, 1]
    tre = p.domain.Time_NRect(1, [-1], [1], timerange)

We then define the boundaries for the domain. We will use periodic boundaries, and so we simply call the function

    bound = p.boundaries.periodic(tre)

As we have a time component, we also must define our initial condition. Derivatives of t has order 1, so we have one initial function.
This is done similarly to our boundaries.

    inits = p.initials.initials(tre, [u0])

We then create the data for the pinn to train on. As we have a time component and we will use an inverse pinn, we create a timeinvpinndata object.
Note that since we use periodic boundaries, no points will be used in training, however we still sample dummy points, so we provide only 
10 to not slow down training.

    dat = p.data.timeinvpinndata(tre, bound, inits, [tdata, xdata], [udata], 12000, 10, 200)

Then, we create the inverse pinn model class and train the model for our desired epochs. When defining our equation, all spatial variables are denoted x1, x2, etc. So our equation is defined as follows. We also then define the constants which will be learned in training.

    eqn = "ut+c*ux1"
    mymodel = p.models.invpinn(dat, [eqn], ["c"])
    mymodel.train(1500)

If we want to quickly visualize our solution and epoch loss, we call the in-built plotting functions for this type of equation. We can also get what the constants trained from the network as well.

    print(mymodel.get_trained_constants())
    p.plotters.plot_solution_prediction_time1D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)

## All Code

    import pinnde as p
    import numpy as np
    import tensorflow as tf

    c = 1
    u0 = lambda x: tf.cos(np.pi*x)
    u_true = lambda x, t: u0(x-c*t)
    N_data = 200
    xdata = np.random.uniform(-1, 1, N_data)
    tdata = np.random.uniform(0, 1, N_data)
    udata = u_true(xdata, tdata)

    timerange = [0, 1]
    tre = p.domain.Time_NRect(1, [-1], [1], timerange)

    bound = p.boundaries.periodic(tre)

    u0func = lambda x1: tf.cos(np.pi*x1)
    inits = p.initials.initials(tre, [u0])

    dat = p.data.timeinvpinndata(tre, bound, inits, [tdata, xdata], [udata], 12000, 10, 200)

    eqn = "ut+c*ux1"
    mymodel = p.models.invpinn(dat, [eqn], ["c"])
    mymodel.train(1500)

    print(mymodel.get_trained_constants())
    p.plotters.plot_solution_prediction_time1D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)

Or more concisely,

    import pinnde as p
    import numpy as np
    import tensorflow as tf

    c = 1
    u0 = lambda x: tf.cos(np.pi*x)
    u_true = lambda x, t: u0(x-c*t)
    N_data = 200
    xdata = np.random.uniform(-1, 1, N_data)
    tdata = np.random.uniform(0, 1, N_data)
    udata = u_true(xdata, tdata)

    tre = p.domain.Time_NRect(1, [-1], [1], [0,1])
    bound = p.boundaries.periodic(tre)
    inits = p.initials.initials(tre, [u0])
    dat = p.data.timeinvpinndata(tre, bound, inits, [tdata, xdata], [udata], 10000, 10, 200)
    mymodel = p.models.invpinn(dat, ["ut+c*ux1"], ["c"])
    mymodel.train(1500)
    print(mymodel.get_trained_constants())
    p.plotters.plot_solution_prediction_time1D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)