# pde_ModelFuncs_2var

Functions for creating PINN's in PDE solvers of tx and xy.

While available to user, not meant to be used. Instead interface through
net_layers, net_units, and constraint parameters of solving functions detailed
in "Main User Functions".

We have described commonly used layer classes used throughout many models, models for specific solvePDE_tx
equations, and models for specifc solvePDE_xy equations.

**Commonly used layer classes**

::: src.pinnDE.PDE.ModelFuncs.pde_ModelFuncs_2var
    options:
        members:
          - Periodic
          - Normalize
          - select_model_tx
          - build_model_standard
          - build_model_periodic_tx
          - build_model_periodic_hardconstraint1_tx
          - build_model_periodic_hardconstraint2_tx
          - build_model_periodic_hardconstraint3_tx
          - select_model_xy
          - build_model_standard
          - build_model_periodic_xy
          - build_model_hardconstraint_xy
    rendering:
      show_root_heading: yes