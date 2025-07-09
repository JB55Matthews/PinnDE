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

* **Inverse PINNs**
    * Solving inverse problems for constants on any problem described above

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