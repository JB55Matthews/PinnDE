# Solving the Poisson equation (#2) with Dirichlet boundries

## Problem
We will look at solving a specific instance of the Poisson equation

$$\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = -(0.4\pi^2\sin(2\pi x) + \frac{200\tanh(10x)}{\cosh(10x)^2})
\sin(2\pi y) $$

$$- 4\pi^2(0.1\sin(2\pi x) + \tanh(10x))\sin(2\pi y)$$

Over $x\in[-1,1], y\in[-1,1]$, with boundary conditions

$$u(x, -1) = u(x, 1) = u(-1, y) = u(1, y) = (0.1\sin(2\pi x) + \tanh(10x))\sin(2\pi y)$$

## Implementation

First import package. We will only import pde_Solvers and pde_Boundries_2var from PinnDE as that is all that is needed. Since we need to represent pi for the equation, we also import numpy. We also use tensorflow's trig funcs (we could not use numpy in this instance as with hard constraints we need tensorflows).

    import pinnde.pde_Solvers as pde_Solvers
    import pinnde.pde_Boundries_2var as pde_Boundries_2var
    import numpy as np
    import tensorflow as tf

We then setup the boundries. We just declare the x boundary, y boundary, number of points along bondary, and declare the 
boundary conditions as a python lambda function which **must** be of 2 variable in a time independent equation

    x_bdry = [-1, 1]
    y_bdry = [-1, 1]
    N_bc = 200
    all_boundry = (0.1*tf.sin(2*np.pi*x) + tf.math.tanh(10*x))*tf.sin(2*np.pi*y)

    boundries = pde_Boundries.setup_boundries_dirichlet_xy(x_bdry, y_bdry, N_bc, 
                                                        all_boundries_cond=all_boundry)

Next, we declare our equation, number of points, and epochs. Equation must be in form eqn = 0

    eqn = "uxx + uyy - (-(0.4*np.pi**2*tf.sin(2*np.pi*x)+200*tf.math.tanh(10*x)/tf.math.cosh(10*x)**2)
    *tf.sin(2*np.pi*y) - 4*np.pi**2*(0.1*tf.sin(2*np.pi*x) + tf.math.tanh(10*x))*tf.sin(2*np.pi*y))"
    N_pde = 10000
    epochs = 2500

To solve, we simply call the corresponding solving function to our problem

    mymodel = pde_Solvers.solvePDE_xy(eqn, boundries, x_bdry, y_bdry, N_pde, epochs)

If we want to quickly vizualize our data from training we can add after the solving function

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()

## All Code

    import pinnde.pde_Solvers as pde_Solvers
    import pinnde.pde_Boundries_2var as pde_Boundries_2var
    import numpy as np
    import tensorflow as tf

    x_bdry = [-1, 1]
    y_bdry = [-1, 1]
    N_bc = 200
    all_boundry = (0.1*tf.sin(2*np.pi*x) + tf.math.tanh(10*x))*tf.sin(2*np.pi*y)

    boundries = pde_Boundries.setup_boundries_dirichlet_xy(x_bdry, y_bdry, N_bc, 
                                                    all_boundries_cond=all_boundry)

    eqn = "uxx + uyy - (-(0.4*np.pi**2*tf.sin(2*np.pi*x)+200*tf.math.tanh(10*x)/tf.math.cosh(10*x)**2)*
    tf.sin(2*np.pi*y) - 4*np.pi**2*(0.1*tf.sin(2*np.pi*x) + tf.math.tanh(10*x))*tf.sin(2*np.pi*y))"
    N_pde = 10000
    epochs = 2500

    mymodel = pde_Solvers.solvePDE_xy(eqn, boundries, x_bdry, y_bdry, N_pde, epochs)

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()
