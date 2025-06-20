# Solving a system of 1+1 pdes with a PINN

## Problem
We will look at solving the PDEs

$$\frac{\partial (u1)}{\partial t} +  \frac{\partial (u1)}{\partial x}= 0$$

$$\frac{\partial (u2)}{\partial t} + (u2)\frac{\partial (u2)}{\partial x} = (\frac{0.01}{\pi})\frac{\partial^2 (u2)}{\partial x^2} $$


Over $x\in[-1,1]$, with initial conditions

$$u1(x, 0) = \cos(\pi x)$$

$$u2(x, 0) = -\sin(\pi x)$$

## Implementation
First import package. Since we need to represent pi for the initial condition, we also import numpy. We also use tensorflow's cos and sin.
    
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

As we have a time component, we also must define our initial condition. Derivatives of t has order 1 in both equations, 
so we have one initial function. This is done similarly to our boundaries.

    u10func = lambda x1: tf.cos(np.pi*x1)
    u20func = lambda x1: -tf.sin(np.pi*x1)
    inits = p.initials.initials(tre, [[u10func], [u20func]])

We then create the data for the pinn to train on. As we have a time component and we will use a pinn, we create a timepinndata object.
Note that since we use periodic boundaries, no points will be used in training, however we still sample dummy points, so we provide only 
10 to not slow down training.

    dat = p.data.timepinndata(tre, bound, inits, 12000, 10, 400)

Then, we create the pinn model class and train the model for our desired epochs. When defining our equation, all spatial variables are denoted
x1, x2, etc. So our equation is defined as follows. We also show how to define the internal layers and nodes to use.

    eqn1 = "u1t+u1x1"
    eqn2 = "u2t+u2*u2x1-(0.01/np.pi)*u2x1x1"
    mymodel = p.models.pinn(dat, [eqn1, eqn2])
    mymodel.train(2500)

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

    u10func = lambda x1: tf.cos(np.pi*x1)
    u20func = lambda x1: -tf.sin(np.pi*x1)
    inits = p.initials.initials(tre, [[u10func], [u20func]])

    dat = p.data.timepinndata(tre, bound, inits, 12000, 10, 400)

    eqn1 = "u1t+u1x1"
    eqn2 = "u2t+u2*u2x1-(0.01/np.pi)*u2x1x1"
    mymodel = p.models.pinn(dat, [eqn1, eqn2])
    mymodel.train(2500)

    p.plotters.plot_solution_prediction_time1D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)

Or more concisely,

    import pinnde as p
    import numpy as np
    import tensorflow as tf

     tre = p.domain.Time_NRect(1, [-1], [1], [0, 1])
    bound = p.boundaries.periodic(tre)
    inits = p.initials.initials(tre, [[lambda x1: tf.cos(np.pi*x1)], [lambda x1: -tf.sin(np.pi*x1)]])
    dat = p.data.timepinndata(tre, bound, inits, 12000, 10, 800)
    mymodel = p.models.pinn(dat, ["u1t+u1x1", "u2t+u2*u2x1-(0.01/np.pi)*u2x1x1"])
    mymodel.train(2500)
    p.plotters.plot_epoch_loss(mymodel)
    p.plotters.plot_solution_prediction_time1D(mymodel)

