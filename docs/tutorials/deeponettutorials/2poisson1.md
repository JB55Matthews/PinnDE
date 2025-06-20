# Solving the Poisson equation with Dirichlet boundaries with a DeepONet

## Problem
We will look at solving a specific instance of the Poisson equation

$$\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = -2\pi^2\cos(\pi x)\sin(\pi y)$$

Over $x\in[-1,1], y\in[-1,1]$, with boundary conditions

$$u(x, -1) = u(x, 1) = u(-1, y) = u(1, y) = \cos(\pi x)\sin(\pi y)$$

## Implementation
First import package. Since we need to represent pi for the boundary condition, we also import numpy. We also use tensorflow's cos.
    
    import pinnde as p
    import numpy as np
    import tensorflow as tf

We then first create our domain. We will solve this on the square from -1 to 1, so we create a NRect with 2 spatial dimensions, with xmins 
[-1, -1] and xmaxs [1, 1]. 

    re = p.domain.NRect(2, [-1, -1], [1, 1])

We then define the boundaries for the domain. We will use Dirichlet boundaries set to zero. Note our lambda function
must be of all dimensions, x1 and x2.

    bdryfunc = lambda x1, x2: tf.cos(np.pi*x1)*tf.sin(np.pi*x2)
    bound = p.boundaries.dirichlet(re, [bdryfunc])

We then create the data for the deeponet to train on. As we have no time component and we will use a deeponet, we create a deeponetdata 
object.

    dat = p.data.dondata(re, bound, 12000, 1000, 2000)

Then, we create the deeponet model class and train the model for our desired epochs. When defining our equation, all spatial variables are denoted
x1, x2, etc. So our equation is defined as follows.

    eqn = "ux1x1 + ux2x2 - (-2*np.pi**2*tf.cos(np.pi*x1)*tf.sin(np.pi*x2))"
    mymodel = p.models.deeponet(dat, [eqn])
    mymodel.train(1500)

If we want to quickly visualize our solution and epoch loss, we call the in-built plotting functions for this type of equation.

    p.plotters.plot_solution_prediction_2D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)

## All Code

    import pinnde as p
    import numpy as np
    import tensorflow as tf

    re = p.domain.NRect(2, [-1, -1], [1, 1])

    bdryfunc = lambda x1, x2: tf.cos(np.pi*x1)*tf.sin(np.pi*x2)
    bound = p.boundaries.dirichlet(re, [bdryfunc])

    dat = p.data.dondata(re, bound, 12000, 1000, 2000)

    eqn = "ux1x1 + ux2x2 - (-2*np.pi**2*tf.cos(np.pi*x1)*tf.sin(np.pi*x2))"
    mymodel = p.models.deeponet(dat, [eqn])
    mymodel.train(1500)

    p.plotters.plot_solution_prediction_2D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)

Or more concisely,

    import pinnde as p
    import numpy as np
    import tensorflow as tf

    re = p.domain.NRect(2, [-1, -1], [1, 1])
    bound = p.boundaries.dirichlet(re, [lambda x1, x2: tf.cos(np.pi*x1)*tf.sin(np.pi*x2)])
    dat = p.data.dondata(re, bound, 12000, 1000, 2000)
    mymodel = p.models.deeponet(dat, ["ux1x1 + ux2x2 - (-2*np.pi**2*tf.cos(np.pi*x1)*tf.sin(np.pi*x2))"])
    mymodel.train(1500)
    p.plotters.plot_solution_prediction_2D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)