# Solving the Poisson equation with Dirichlet boundaries with a DeepONet

## Problem
We will look at solving a specific instance of the Helmholtz equation

$$\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + 4\pi^2\sin(2\pi x)\sin(2\pi y)
= -4\pi^2 u $$

Over $x\in[0,1], y\in[0,1]$, with boundary conditions

$$u(x, 0) = u(x, 1) = u(0, y) = u(1, y) = 0 $$

## Implementation
First import package. Since we need to represent pi for the equation, we also import numpy. We also use tensorflow's cos.
    
    import pinnde as p
    import numpy as np
    import tensorflow as tf

We then first create our domain. We will solve this on the square from 0 to 1, so we create a NRect with 2 spatial dimensions, with xmins 
[0, 0] and xmaxs [1, 1]. 

    re = p.domain.NRect(2, [0, 0], [1, 1])

We then define the boundaries for the domain. We will use Dirichlet boundaries set to zero. Note our lambda function
must be of all dimensions, x1 and x2.

    bdryfunc = lambda x1, x2: 0+x1*0
    bound = p.boundaries.dirichlet(re, [bdryfunc])

We then create the data for the deeponet to train on. As we have no time component and we will use a deeponet, we create a dondata object.

    dat = p.data.dondata(re, bound, 12000, 800, 1000)

Then, we create the deeponet model class and train the model for our desired epochs. When defining our equation, all spatial variables are denoted
x1, x2, etc. So our equation is defined as follows.

    eqn = "ux1x1 + ux2x2 + ((np.pi*2)**2)*u + ((np.pi*2)**2)*tf.sin((np.pi*2)*x1)*tf.sin((np.pi*2)*x2)"
    mymodel = p.models.deeponet(dat, [eqn])
    mymodel.train(2500)

If we want to quickly visualize our solution and epoch loss, we call the in-built plotting functions for this type of equation.

    p.plotters.plot_solution_prediction_2D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)

## All Code

    import pinnde as p
    import numpy as np
    import tensorflow as tf

    re = p.domain.NRect(2, [0, 0], [1, 1])

    bdryfunc = lambda x1, x2: 0+x1*0
    bound = p.boundaries.dirichlet(re, [bdryfunc])

    dat = p.data.dondata(re, bound, 12000, 800, 1000)

    eqn = "ux1x1 + ux2x2 + ((np.pi*2)**2)*u + ((np.pi*2)**2)*tf.sin((np.pi*2)*x1)*tf.sin((np.pi*2)*x2)"
    mymodel = p.models.deeponet(dat, [eqn])
    mymodel.train(2500)

    p.plotters.plot_solution_prediction_2D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)

Or more concisely,

    import pinnde as p
    import numpy as np
    import tensorflow as tf

    re = p.domain.NRect(2, [0, 0], [1, 1])
    bound = p.boundaries.dirichlet(re, [lambda x1, x2: 0+x1*0])
    dat = p.data.dondata(re, bound, 12000, 800, 1000)
    mymodel = p.models.deeponet(dat, ["ux1x1 + ux2x2 + ((np.pi*2)**2)*u + ((np.pi*2)**2)*tf.sin((np.pi*2)*x1)*tf.sin((np.pi*2)*x2)"])
    mymodel.train(2500)
    p.plotters.plot_solution_prediction_2D(mymodel)
    p.plotters.plot_epoch_loss(mymodel)