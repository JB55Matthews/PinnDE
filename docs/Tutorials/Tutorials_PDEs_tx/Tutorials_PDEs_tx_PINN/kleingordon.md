# Solving the Klein-Gorodn equation, specifying the PINN architecture

## Problem
We will look at solving a specific instance of the Klein-Gordon equation

$$\frac{\partial^2 u}{\partial t^2} - \frac{\partial^2 u}{\partial x^2} + u^2 = x^2 \cos(t)^2 - x\cos(t) $$

Over $t\in[0,1], x\in[-1,1]$, with initial conditions

$$u(x, 0) = x, \frac{\partial u}{\partial t}(x, 0) = 0$$


With our dirichlet boundary conditions defined as

$$u(-1, t) = -\cos(t) , u(1, t) = \cos(t)$$

Exact solution: 

$$x\cos(t)$$

## Implementation

First import package. We will only import pde_Solvers, pde_Initials and pde_Boundaries_2var from PinnDE as that is all that is needed. We also use tensorflow's cos (we could use numpy in this instance, however with hard constraints we need tensorflows so we use this always)

    import pinnde.pde_Solvers as pde_Solvers
    import pinnde.pde_Initials as pde_Initials
    import pinnde.pde_Boundaries_2var as pde_Boundaries
    import tensorflow as tf

We first can create our initial conditions as a python lambda functions. They **must** be of 1 variable in a time dependent equation.
We declare t boundary, x_boundary, the order of t, and number of points along initial t. Note we still use variable to make condition a constant.

    u0 = lambda x: x
    ut0 = lambda x: 0+x*0
    t_bdry = [0, 10]
    x_bdry = [-1, 1]
    t_order = 2
    N_iv = 200
    initials = pde_Initials.setup_initials_2var(t_bdry, x_bdry, t_order, [u0, ut0], N_iv)

We then setup the boundaries. We just declare the number of points along bondary, and declare the 
boundary conditions as a python lambda function which **must** be of 1 variable in a time dependent equation.

    N_bc = 100
    xleft_boundary = lambda x: -(tf.cos(x))
    xright_boundary = lambda x: tf.cos(x)
    boundaries = pde_Boundaries.setup_boundaries_dirichlet_tx(t_bdry, x_bdry, N_bc, 
                            xleft_boundary_cond=xleft_boundary, xright_boundary_cond=xright_boundary)

Next, we declare our equation, number of points, and epochs. 
Equation must be in form eqn = 0

    eqn = "utt - uxx + u**2 - (-(x*tf.cos(t)) + (x**2)*((tf.cos(t))**2))"
    N_pde = 10000
    epochs = 3000

If we also want to change the default number of internal layers (4) and nodes (60) per layer in our PINN, we can declare them as well

    layers = 6
    nodes = 50

To solve, we simply call the corresponding solving function to our problem, and train the model

    mymodel = pde_Solvers.solvePDE_tx(eqn, initials, boundaries, N_pde, 
                                        net_layers = layers, net_units = units)
    mymodel.train_model(epochs)

If we want to quickly vizualize our data from training we can add after the solving function

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()

## All Code

    import pinnde.pde_Solvers as pde_Solvers
    import pinnde.pde_Initials as pde_Initials
    import pinnde.pde_Boundaries_2var as pde_Boundaries
    import tensorflow as tf

    u0 = lambda x: x
    ut0 = lambda x: 0+x*0
    t_bdry = [0, 10]
    x_bdry = [-1, 1]
    t_order = 2
    N_iv = 200
    initials = pde_Initials.setup_initials_2var(t_bdry, x_bdry, t_order, [u0, ut0], N_iv)

    N_bc = 100
    xleft_boundary = lambda x: -(tf.cos(x))
    xright_boundary = lambda x: tf.cos(x)
    boundaries = pde_Boundaries.setup_boundaries_dirichlet_tx(t_bdry, x_bdry, N_bc, 
                            xleft_boundary_cond=xleft_boundary, xright_boundary_cond=xright_boundary)

    eqn = "utt - uxx + u**2 - (-(x*tf.cos(t)) + (x**2)*((tf.cos(t))**2))"
    N_pde = 10000
    epochs = 3000
    layers = 6
    nodes = 50

    mymodel = pde_Solvers.solvePDE_tx(eqn, initials, boundaries, N_pde, 
                                        net_layers = layers, net_units = nodes)
    mymodel.train_model(epochs)

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()