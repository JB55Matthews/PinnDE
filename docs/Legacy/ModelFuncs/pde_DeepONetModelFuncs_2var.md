# pde_DeepONetModelFuncs_2var

Functions for creating PINN's in PDE DeepONet solvers of tx and xy.

While available to user, not meant to be used. Instead interface through
net_layers, net_units, and constraint parameters of solving functions detailed
in "Main User Functions".

We have described commonly used layer classes and functions used throughout many models, models for specific 
solvePDE_DeepONet_tx equations, and models for specifc solvePDE_DeepONet_xy equations.

**Commonly used layer classes and functions**

::: src.pinnde.legacy.PDE.ModelFuncs.pde_DeepONetModelFuncs_2var
    options:
        members:
            - PeriodicBCs
            - Normalize
            - mlp_network
            - select_DeepONet_tx
            - build_DeepONet_standard
            - build_DeepONet_periodic_tx
            - build_DeepONet_periodic_tx_hard1
            - build_DeepONet_periodic_tx_hard2
            - build_DeepONet_periodic_tx_hard3
            - select_DeepONet_xy
            - build_DeepONet_standard
            - build_DeepONet_periodic_xy
            - build_DeepONet_dirichlet_hardconstraint_xy
    rendering:
      show_root_heading: yes