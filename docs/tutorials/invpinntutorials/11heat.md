# Solving for heat diffusion the 1+1 heat equation with Dirichlet boundaries with an Inverse PINN

## Problem
We will look at solving for the heat diffusion h of the Heat equation

$$ h\frac{\partial^2 u}{\partial x^2}= \frac{\partial u}{\partial t} $$

Over $t\in[0,1]$, $x\in[0, 1]$

$$u(x, 0) = \sin(\pi x)$$

## Implementation
First import package. Since we need to represent pi for the initial condition, we also import numpy. We also use tensorflow's sin.
    
    import pinnde as p
    import numpy as np
    import tensorflow as tf

We first then generate our sample data. We have the reference solution to this problem, with h=0.08, and so we generate data along t, x1, 
and for the reference solution u to mimic having experimental data.

    h = 0.08
    u_true = lambda x, t: np.e**(-((np.pi**2) * h * t))*np.sin(np.pi*x)
    N_data = 400
    xdata = np.random.uniform(0, 1, N_data)
    tdata = np.random.uniform(0, 1, N_data)
    udata = u_true(xdata, tdata)

We then create our domain. We will solve this on the interval 0 to 1, and this equation has a time component, so
we create a Time_NRect with 1 spatial dimension. We will solve from time t=0 to t=1.

    timerange = [0, 1]
    tre = p.domain.Time_NRect(1, [0], [1], timerange)

We then define the boundaries for the domain. We will use Dirichlet boundaries set to zero. Note our lambda function
must be of all dimensions, t, and x1, and one of them must be used in the function. Setting a constant is done as follows.

    bdryfunc = lambda t, x1: 0+t*0
    bound = p.boundaries.dirichlet(tre, [bdryfunc])

As we have a time component, we also must define our initial condition. Derivatives of t has order 1, so we have one initial function.
This is done similarly to our boundaries.

    u0func = lambda x1: tf.sin(np.pi*x1)
    inits = p.initials.initials(dom, [u0func])

We then create the data for the pinn to train on. As we have a time component and we will use an inverse pinn, we create a timeinvpinndata object.

    dat = p.data.timeinvpinndata(dom, bound, inits, [tdata, xdata], [udata], 12000, 1000, 1000)

Then, we create the inverse pinn model class and train the model for our desired epochs. When defining our equation, all spatial variables are denoted x1, x2, etc. So our equation is defined as follows.

    eqn = "h*ux1x1 - ut"
    mymodel = p.models.invpinn(dat, [eqn], ["h"])
    mymodel.train(2000)

If we want to quickly visualize our solution and epoch loss, we call the in-built plotting functions for this type of equation. We can also get what the constants trained from the network as well.

    print(mymodel.get_trained_constants())
    p.plotters.plot_solution_prediction_time1D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)

## All Code

    import pinnde as p
    import numpy as np
    import tensorflow as tf

    h = 0.08
    u_true = lambda x, t: np.e**(-((np.pi**2) * h * t))*np.sin(np.pi*x)
    N_data = 400
    xdata = np.random.uniform(0, 1, N_data)
    tdata = np.random.uniform(0, 1, N_data)
    udata = u_true(xdata, tdata)

    timerange = [0, 1]
    tre = p.domain.Time_NRect(1, [0], [1], timerange)

    bdryfunc = lambda t, x1: 0+t*0
    bound = p.boundaries.dirichlet(tre, [bdryfunc])

    u0func = lambda x1: tf.sin(np.pi*x1)
    inits = p.initials.initials(dom, [u0func])

    dat = p.data.timeinvpinndata(dom, bound, inits, [tdata, xdata], [udata], 12000, 1000, 1000)

    eqn = "h*ux1x1 - ut"
    mymodel = p.models.invpinn(dat, [eqn], ["h"])
    mymodel.train(2000)

    print(mymodel.get_trained_constants())
    p.plotters.plot_solution_prediction_time1D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)

Or more concisely,

    import pinnde as p
    import numpy as np
    import tensorflow as tf

    h = 0.08
    u_true = lambda x, t: np.e**(-((np.pi**2) * h * t))*np.sin(np.pi*x)
    N_data = 400
    xdata = np.random.uniform(0, 1, N_data)
    tdata = np.random.uniform(0, 1, N_data)
    udata = u_true(xdata, tdata)

    tre = p.domain.Time_NRect(1, [0], [1], [0,1])
    bound = p.boundaries.dirichlet(tre, [lambda t, x1: 0+t*0 ])
    inits = p.initials.initials(tre, [lambda x1: tf.sin(np.pi*x1)])
    dat = p.data.timeinvpinndata(tre, bound, inits, [tdata, xdata], [udata], 12000, 1000, 1000)
    mymodel = p.models.invpinn(dat, ["h*ux1x1 - ut"], ["h"])
    mymodel.train(2000)
    print(mymodel.get_trained_constants())
    p.plotters.plot_solution_prediction_time1D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)

