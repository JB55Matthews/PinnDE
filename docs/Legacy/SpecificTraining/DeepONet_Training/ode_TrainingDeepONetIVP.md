# ode_TrainingDeepONetIVP

Functions which trains a DeepONet for solving a solveODE_DeepONet_IVP calls implemented in JAX/FLAX

Each of order 1,2, and 3 equations with hard and soft constraints have there own training file respectively. Almost all functions are same, with slight changes made to model and model-interacting functions for specific problems. Here we outline all these commonly used functions. All files can be found in github in ODE/SpecificTraining/DeepONet/DeepONetIVP directory.

While available to user, not meant to be used. Meant to be used through
object returned from solveODE calls, where training file is selected through ode_trainingSelect

::: src.pinnde.legacy.ODE.SpecificTraining.DeepONet_Training.DeepONet_IVP.documentation_TrainingDeepONetIVP
    options:
        members:
            - startTraining
            - Normalize
            - CombineBranches
            - HardConstraint
            - MLP
            - DeepONet
            - train_network
            - train_step
            - defineCollocationPoints
        group_by_category: no
    rendering:
      show_root_heading: yes