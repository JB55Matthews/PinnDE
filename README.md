PinnDE
--------

Physics Informed Neural Networks for Differential Equations (PinnDE) is an open-source library
in Python 3 for solving ordinary and partial differential equations (ODEs and PDEs) using both
physics informed neural networks (PINNs) and deep operator networks (DeepONets). The goal of PinnDE is to
provide a user-friendly library as an alternative to the more powerful but more complex alternatives. This library provides
simple, user-friendly interfacing of solving methods which can easily be used in collaboration with 
non-profficent users of python or the library, where collaborators should be able to understand the contents of
the code quickly and without having to learn the library themselves.

The documentation can be found [here](https://pinnde.readthedocs.io/en/latest/)

Installation
----------
This package requires *numpy*, *tensorflow*, *jax/flax/optax*, *matplotlib*, and *pyDOE*. These 
are all installed with the package. If version of a package already installed which is above the reqiements
for PinnDE, then currently package won't be upgraded when installed.

Installing can simply be done with pip in the command line with

    pip install pinnde

Citing
--------
# Citing

If PinnDE is used in academic research, please cite the paper found [here](https://arxiv.org/abs/2408.10011), 
or with the corresponding BibTex citation

    @misc{matthews2024pinndephysicsinformedneuralnetworks,
        title={PinnDE: Physics-Informed Neural Networks for Solving Differential Equations}, 
        author={Jason Matthews and Alex Bihlo},
        year={2024},
        eprint={2408.10011},
        archivePrefix={arXiv},
        primaryClass={cs.LG},
        url={https://arxiv.org/abs/2408.10011}, 
    }
