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

First import package. We will only import pde_Solvers and pde_Boundries_2var from pinnDE as that is all that is needed. We also use tensorflow's cos (we could use numpy in this instance, however with hard constraints we need tensorflows so we use this always)

    import pinnDE.pde_Solvers as pde_Solvers
    import pinnDE.pde_Boundries_2var as pde_Boundries_2var
    import tensorflow as tf

We first can create our initial conditions as a python lambda functions. They **must** be of 1 variable in a time dependent equation

    u0 = lambda x: x
    ut0 = lambda x: 0+x*0

We then setup the boundries. We just declare the t boundary, x boundary, number of points along bondary, and declare the 
boundary conditions as a python lambda function which **must** be of 1 variable in a time dependent equation. Note
how we must use the vairbale declared even just for making the boundries a constant

    t_bdry = [0, 1]
    x_bdry = [-1, 1]
    N_bc = 100
    xleft_boundry = lambda x: -(tf.cos(x))
    xright_boundry = lambda x: tf.cos(x)
    boundries = pde_Boundries.setup_boundries_dirichlet_tx(t_bdry, x_bdry, N_bc, 
                            xleft_boundry_cond=xleft_boundry, xright_boundry_cond=xright_boundry)

Next, we declare our equation, order of t, initial condition, number of points, number of inital value points, and epochs. 
Equation must be in form eqn = 0

    eqn = "utt - uxx + u**2 - (-(x*tf.cos(t)) + (x**2)*((tf.cos(t))**2))"
    t_order = 2
    initial_cond = [u0, ut0]
    N_pde = 10000
    N_iv = 200
    epochs = 3000

If we also want to change the default number of internal layers (4) and nodes (60) per layer in our PINN, we can declare them as well

    layers = 6
    nodes = 50

To solve, we simply call the corresponding solving function to our problem

    mymodel = pde_Solvers.solvePDE_tx(eqn, t_order, initial_cond, boundries, t_bdry, x_bdry, 
                                        N_pde, N_iv, epochs, net_layers = layers, net_units = units)

If we want to quickly vizualize our data from training we can add after the solving function

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()

## All Code

    import pinnDE.pde_Solvers as pde_Solvers
    import pinnDE.pde_Boundries_2var as pde_Boundries_2var
    import tensorflow as tf

    u0 = lambda x: x
    ut0 = lambda x: 0+x*0

    t_bdry = [0, 1]
    x_bdry = [-1, 1]
    N_bc = 100
    xleft_boundry = lambda x: -(tf.cos(x))
    xright_boundry = lambda x: tf.cos(x)
    boundries = pde_Boundries.setup_boundries_dirichlet_tx(t_bdry, x_bdry, N_bc, 
                            xleft_boundry_cond=xleft_boundry, xright_boundry_cond=xright_boundry)

    eqn = "utt - uxx + u**2 - (-(x*tf.cos(t)) + (x**2)*((tf.cos(t))**2))"
    t_order = 2
    initial_cond = [u0, ut0]
    N_pde = 10000
    N_iv = 200
    epochs = 3000
    layers = 6
    nodes = 50

    mymodel = pde_Solvers.solvePDE_tx(eqn, t_order, initial_cond, boundries, t_bdry, x_bdry, 
                                        N_pde, N_iv, epochs, net_layers = layers, net_units = units)

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()