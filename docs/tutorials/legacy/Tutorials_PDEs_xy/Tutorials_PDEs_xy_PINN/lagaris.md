# Solving Problem 5 in Lagaris with Hard constrainted Dirichlet boundaries

## Problem
We will look at solving problem 5 found in [Lagaris et al.](https://arxiv.org/abs/physics/9705023), which demonstrates
how to hard constraint dirichlet boundary conditions

$$\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = e^{-x}(x-2+y^3+6y) $$

Over $x\in[0,1], y\in[0,1]$, with boundary conditions

$$u(0, y) = y^3,  u(1, y) = (1+y^3)e^{-1} $$

$$u(x, 0) = xe^{-x}, u(x, 1) = e^{-x}(x+1)$$

The analytic solution is $e^{-x}(x + y^3)$

## Implementation

First import package. We will only import pde_Solvers and pde_Boundaries_2var from PinnDE as that is all that is needed. Since we need to represent e for the equation, we also import numpy.

    import pinnde.pde_Solvers as pde_Solvers
    import pinnde.pde_Boundaries_2var as pde_Boundaries_2var
    import numpy as np

We then setup the boundaries. We just declare the x boundary, y boundary, number of points along bondary, and declare the 
boundary conditions as a python lambda function which **must** be of 2 variable in a time independent equation.

    x_bdry = [0, 1]
    y_bdry = [0, 1]
    N_bc = 400
    x_left = lambda x, y: y**3
    x_right = lambda x, y: (1+y**3)/(np.e)
    y_lower = lambda x, y: x/(np.e**x)
    y_upper = lambda x, y: (x+1)/(np.e**x)

    boundaries = pde_Boundaries_2var.setup_boundaries_dirichlet_xy(x_bdry,y_bdry,N_bc, xleft_boundary_cond=x_left,
                    xright_boundary_cond=x_right, ylower_boundary_cond=y_lower, yupper_upper_cond=y_upper)

Next, we declare our equation, number of points, and epochs. Equation must be in form eqn = 0

    eqn = "uxx+uyy - (x-2+6*y+y**3)/(np.e**x)"
    N_pde = 10000
    epochs = 1000

To solve, we simply call the corresponding solving function to our problem, declaring a hard constraint, and train the model

    mymodel = pde_Solvers.solvePDE_xy(eqn, boundaries, N_pde, constraint = "hard")
    mymodel.train_model(epochs)

If we want to quickly vizualize our data from training we can add after the solving function

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()

## All Code

    import pinnde.pde_Solvers as pde_Solvers
    import pinnde.pde_Boundaries_2var as pde_Boundaries_2var
    import numpy as np

    x_bdry = [0, 1]
    y_bdry = [0, 1]
    N_bc = 400
    x_left = lambda x, y: y**3
    x_right = lambda x, y: (1+y**3)/(np.e)
    y_lower = lambda x, y: x/(np.e**x)
    y_upper = lambda x, y: (x+1)/(np.e**x)

    boundaries = pde_Boundaries_2var.setup_boundaries_dirichlet_xy(x_bdry,y_bdry,N_bc, xleft_boundary_cond=x_left,
                xright_boundary_cond=x_right, ylower_boundary_cond=y_lower, yupper_boundary_cond=y_upper)

    eqn = "uxx+uyy - (x-2+6*y+y**3)/(np.e**x)"
    N_pde = 10000
    epochs = 1000

    mymodel = pde_Solvers.solvePDE_xy(eqn, boundaries, N_pde, constraint = "hard")
    mymodel.train_model(epochs)

    mymodel.plot_epoch_loss()

    mymodel.plot_solution_prediction()