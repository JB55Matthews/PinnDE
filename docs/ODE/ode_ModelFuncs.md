# ode_ModelFuncs

Functions for creating PINN's in ODE solvers.

While available to user, not meant to be used. Instead interface through
net_layers, net_units, and constraint parameters of solving functions detailed
in "Main User Functions".

::: src.pinnDE.ODE.ode_ModelFuncs
    options:
        members_order: source
    rendering:
      show_root_heading: yes
    selection:
      members:
        - build_model
        - build_model_hardConstraint_order1_IVP
        - build_model_hardConstraint_order2_IVP
        - build_model_hardConstraint_order3_IVP
        - build_model_hardConstraint_order4_IVP
        - build_model_hardConstraint_order5_IVP
        - build_model_hardConstraint_order12_BVP
        - build_model_hardConstraint_order3_BVP
        - build_model_system2_IVP
        - build_model_system3_IVP