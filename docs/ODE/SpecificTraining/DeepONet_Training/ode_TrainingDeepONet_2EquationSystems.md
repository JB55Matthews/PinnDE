# ode_TrainingDeepONet_2EquationSystems

Functions which trains a DeepONet for solving a solveODE_DeepONetSystem_IVP calls of 2 equations, implemented in JAX/FLAX

Each combination of 1, 2, and 3 order equations with hard constraints only have there own training file respectively. Almost all functions are the same, with slight changes made to model and model-interacting functions for specific problems. Here we outline all these commonly used functions. All files can be found in github in ODE/SpecificTraining/DeepONet/DeepONet_2EquationSystems directory.

While available to user, not meant to be used. Meant to be used through
object returned from solveODE calls, where training file is selected through ode_trainingSelect

::: src.pinnde.ODE.SpecificTraining.DeepONet_Training.DeepONet_2EquationSystems.documentation_TrainingDeepONet_2EquationSystems
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