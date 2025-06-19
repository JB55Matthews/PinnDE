# Welcome to PinnDE!

Physics Informed Neural Networks for Differential Equations (PinnDE) is an open-source library
in Python 3 for solving ordinary and partial differential equations (ODEs and PDEs) using both
physics informed neural networks (PINNs) and deep operator networks (DeepONets). The goal of PinnDE is to
provide a user-friendly library as an alternative to the more powerful but more complex alternative packages
that are available within this field. This library provides simple, user-friendly interfacing of solving methods which can easily be used in collaboration with 
non-proficient users of python or the library, where collaborators should be able to understand the contents of
the code quickly and without having to learn the library themselves. We also propose the use of PinnDE for education use.
Methods in this field may be taught by educators at a low level may be understandable to students, but the code to implement
these ideas can be large and more difficult to grasp. PinnDE provides simple implementations where students can experiment with different
variations of model parameters and training methods without needing to delve into low level implementations.

We currently provide implementations to solve the following problems

PINNs
----
* Arbitrary order ODEs/PDEs
* Arbitrary Systems of ODEs/PDEs
* Domains
    * Generalized N dimensional hyperrectangles
    * Generalized N dimensional ellipsoids
* Solve spatio-temporal equations (1+n)
* Solve purely spatial equations
* Simple to use plotting modules for 1+1, 1+2, spatio-temporal equations, and 1 and 2 dimensional spatial equations

DeepONets
-----
* Arbitrary order ODEs/PDEs
* Domains
    * Generalized N dimensional hyperrectangles
    * Generalized N dimensional ellipsoids
* Solve spatio-temporal equations (1+n)
* Solve purely spatial equations
* Simple to use plotting modules for 1+1, 1+2, spatio-temporal equations, and 1 and 2 dimensional spatial equations

We provide Periodic, Dirichlet, and Neumann boundary conditions for each of these PDE problems

### Tutorials
* [ODE Tutorials](Tutorials/Tutorials_ODEs/ODE_Tutorials.md)
* [PDEs in t and x Tutorials](Tutorials/Tutorials_PDEs_tx/PDE_tx_tutorials.md)
* [PDEs in x and y Tutorials](Tutorials/Tutorials_PDEs_xy/PDE_xy_tutorials.md)

<!-- ## Main Functions in Solving API
### Solvers, Boundaries, and Initials
* [ODE Solvers](Legacy/ode_Solvers.md)
* [PDE Solvers](Legacy/pde_Solvers.md)
* [PDE Boundaries](Legacy/pde_Boundaries_2var.md)
* [PDE Initial Conditions](Legacy/pde_Initials.md)
### Classes returned in Solving
* ode Solve Classes
    * [ode_SolveClass](Legacy/odeSolveClasses/ode_SolveClass.md)
    * [ode_SystemSolveClass](Legacy/odeSolveClasses/ode_SystemSolveClass.md)
    * [ode_DeepONetSolveClass](Legacy/odeSolveClasses/ode_DeepONetSolveClass.md)
    * [ode_SystemDeepONetSolveClass](Legacy/odeSolveClasses/ode_SystemDeepONetSolveClass.md)

* pde Solve Classes:
    * [pde_SolveClass_tx](Legacy/pdeSolveClasses/pde_SolveClass_tx.md)
    * [pde_SolveClass_xy](Legacy/pdeSolveClasses/pde_SolveClass_xy.md)
    * [pde_DeepONetSolveClass_tx](Legacy/pdeSolveClasses/pde_DeepONetSolveClass_tx.md)
    * [pde_DeepONetSolveClass_xy](Legacy/pdeSolveClasses/pde_DeepONetSolveClass_xy.md)


Functions in API under ODE and PDE are internally used, organized in structure found in code -->
