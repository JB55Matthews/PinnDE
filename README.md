PinnDE
--------

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

The documentation can be found [here](https://pinnde.readthedocs.io/en/latest/)

Installation
----------
This package requires *numpy*, *tensorflow*, *jax/flax/optax*, *matplotlib*, and *pyDOE*. These 
are all installed with the package. If version of a package already installed which is above the requirements
for PinnDE, then currently package won't be upgraded when installed. 

Installing can simply be done with pip in the command line with

    pip install pinnde

Citing
-------
If PinnDE is used in academic research, please cite the paper found [here](https://arxiv.org/abs/2408.10011), 
or with the corresponding BibTex citation

    @article{matthews2024pinnde,
            title={PinnDE: Physics-Informed Neural Networks for Solving Differential Equations},
            author={Matthews, Jason and Bihlo, Alex},
            journal={arXiv preprint arXiv:2408.10011},
            year={2024}
    }
