# Welcome to PinnDE!

Physics Informed Neural Networks for Differential Equations (PinnDE) is an open-source library
in Python 3 for solving ordinary and partial differential equations (ODEs and PDEs) using both
physics informed neural networks (PINNs) and deep operator networks (DeepONets). The goal of PinnDE is to
provide a user-friendly library as an alternative to the more powerful but more complex alternatives of 
[DeepXDE](https://github.com/lululxvi/deepxde) or [PINA](https://github.com/mathLab/PINA). This library provides
simple, user-friendly interfacing of solving methods which can easily be used in collaboration with 
non-profficent users of python or the library, where collaborators should be able to understand the contents of
the code quickly and without having to learn the library themselves.

## Referneces of Implementations

**PINN:** Our basis of a PINN is based on this [Raissi et al., paper](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)
most commonly used when citing PINN's. This is where the basis of our PINN originates.
We also acknowledge that most hard constrainting methods used in both PINN's and DeepONets
is from what is described in this [Lagaris et al., paper](https://arxiv.org/abs/physics/9705023).

**DeepONet:** Our main basis for a Deep Operator Netork comes from what is described in this
[Lu et al., paper](https://arxiv.org/abs/1910.03193), and from what is described in this
[Wang and Perdikaris paper](https://www.sciencedirect.com/science/article/abs/pii/S0021999122009184).


### Tutorials
* [ODE Tuorials](Tutorials/Tutorials_ODEs/ODE_Tutorials.md)
* [PDEs in t and x Tutorials](Tutorials/Tutorials_PDEs_tx/PDE_tx_tutorials.md)
* [PDEs in x and y Tutorials](Tutorials/Tutorials_PDEs_xy/PDE_xy_tutorials.md)

## Main Functions in Solving API
### Solvers and Boundaries
* [ODE Solvers](MainUserFunctions/ode_Solvers.md)
* [PDE Solvers](MainUserFunctions/pde_Solvers.md)
* [PDE Boundaries](MainUserFunctions/pde_Boundaries_2var.md)
* [PDE Initial Conditions](MainUserFunctions/pde_Initials.md)
### Classes returned in Solving
* ode Solve Classes
    * [ode_SolveClass](MainSolveClasses/odeSolveClasses/ode_SolveClass.md)
    * [ode_SystemSolveClass](MainSolveClasses/odeSolveClasses/ode_SystemSolveClass.md)
    * [ode_DeepONetSolveClass](MainSolveClasses/odeSolveClasses/ode_DeepONetSolveClass.md)
    * [ode_SystemDeepONetSolveClass](MainSolveClasses/odeSolveClasses/ode_SystemDeepONetSolveClass.md)

* pde Solve Classes:
    * [pde_SolveClass_tx](MainSolveClasses/pdeSolveClasses/pde_SolveClass_tx.md)
    * [pde_SolveClass_xy](MainSolveClasses/pdeSolveClasses/pde_SolveClass_xy.md)
    * [pde_DeepONetSolveClass_tx](MainSolveClasses/pdeSolveClasses/pde_DeepONetSolveClass_tx.md)
    * [pde_DeepONetSolveClass_xy](MainSolveClasses/pdeSolveClasses/pde_DeepONetSolveClass_xy.md)


Functions in API under ODE and PDE are internally used, organized in structure found in code