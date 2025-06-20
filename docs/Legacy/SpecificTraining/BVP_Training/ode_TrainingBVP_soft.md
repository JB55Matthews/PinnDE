# ode_TrainingBVP_soft

Functions which trains a model for solving a solveODE_BVP call with soft constraints implemented in TensorFlow.

While available to user, not meant to be used. Meant to be used through
object returned from solveODE calls, where training file is selected through ode_trainingSelect

::: src.pinnde.legacy.ODE.SpecificTraining.BVP_Training.ode_TrainingBVP_soft
    rendering:
      show_root_heading: yes
    selection:
      members:
        - PINNtrain_IVP
        - train_network_BVP