# Solving the 1+2 heat equation over the unit circle with Dirichlet boundaries with a PINN

## Problem
We will look at solving a specific instance of the Heat equation

$$ 0.08\frac{\partial^2 u}{\partial x^2} + 0.08\frac{\partial^2 u}{\partial y^2}= \frac{\partial u}{\partial t} $$

Over $t\in[0,1]$, and the unit circle, with initial condition

$$u(x, y, 0) = \sin(\pi x)\sin(\pi y)$$

## Implementation
First import package. Since we need to represent pi for the initial condition, we also import numpy. We also use tensorflow's sin.
    
    import pinnde as p
    import numpy as np
    import tensorflow as tf

We then first create our domain. We will solve this on the unit circle, and this equation has a time component, so
we create a Time_NEllipsoid with 2 spatial dimensions, with center (0.5, 0.5), and semilengths (0.5, 0.5). We will solve
from time t=0 to t=1.

    center = [0.5, 0.5]
    semilengths = [0.5, 0.5]
    timerange = [0, 1]
    dom = p.domain.Time_NEllipsoid(2, center, semilengths, timerange)

We then define the boundaries for the domain. We will use Dirichlet boundaries set to zero. Note our lambda function
must be of all dimensions, t, x1, and x2, and one of them must be used in the function. Setting a constant is done as follows.

    bdryfunc = lambda t, x1, x2: 0+t*0
    bound = p.boundaries.dirichlet(tre, [bdryfunc])

As we have a time component, we also must define our initial condition. Derivatives of t has order 1, so we have one initial function.
This is done similarly to our boundaries.

    u0func = lambda x1, x2: tf.sin(np.pi*x1)*tf.sin(np.pi*x2)
    inits = p.initials.initials(dom, [u0func])

We then create the data for the pinn to train on. As we have a time component and we will use a pinn, we create a timepinndata object.

    dat = p.data.timepinndata(dom, bound, inits, 12000, 600, 400)

Then, we create the pinn model class and train the model for our desired epochs. When defining our equation, all spatial variables are denoted
x1, x2, etc. So our equation is defined as follows.

    eqn = "0.08*ux1x1 + 0.08*ux2x2 - ut"
    mymodel = p.models.pinn(dat, [eqn])
    mymodel.train(800)

If we want to quickly visualize our solution and epoch loss, we call the in-built plotting functions for this type of equation.

    p.plotters.plot_solution_prediction_time2D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)

## All Code

    import pinnde as p
    import numpy as np
    import tensorflow as tf

    center = [0.5, 0.5]
    semilengths = [0.5, 0.5]
    timerange = [0, 1]
    dom = p.domain.Time_NEllipsoid(2, center, semilengths, timerange)

    bdryfunc = lambda t, x1, x2: 0+t*0
    bound = p.boundaries.dirichlet(tre, [bdryfunc])

    u0func = lambda x1, x2: tf.sin(np.pi*x1)*tf.sin(np.pi*x2)
    inits = p.initials.initials(dom, [u0func])

    dat = p.data.timepinndata(dom, bound, inits, 12000, 600, 400)

    eqn = "0.08*ux1x1 + 0.08*ux2x2 - ut"
    mymodel = p.models.pinn(dat, [eqn])
    mymodel.train(800)

    p.plotters.plot_solution_prediction_time2D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)

Or more concisely,

    import pinnde as p
    import numpy as np
    import tensorflow as tf

    dom = p.domain.Time_NEllipsoid(2, [0.5, 0.5], [0.5, 0.5], [0, 1])
    bound = p.boundaries.dirichlet(dom, [lambda t, x1, x2: 0+t*0 ])
    inits = p.initials.initials(dom, [lambda x1, x2: tf.sin(np.pi*x1)*tf.sin(np.pi*x2)])
    dat = p.data.timepinndata(dom, bound, inits, 12000, 600, 300)
    mymodel = p.models.pinn(dat, ["0.08*ux1x1 + 0.08*ux2x2 - ut"])
    mymodel.train(800)
    p.plotters.plot_solution_prediction_time2D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)

