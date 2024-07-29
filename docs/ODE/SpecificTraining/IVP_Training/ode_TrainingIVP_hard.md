# ode_TrainingIVP_Hard

Functions which trains a model for solving a solveODE_IVP call with hard constraints implemented in TensorFlow.

While available to user, not meant to be used. Meant to be used through
object returned from solveODE calls, where training file is selected through ode_trainingSelect

::: src.pinnDE.ODE.SpecificTraining.IVP_Training.ode_TrainingIVP_Hard
    rendering:
      show_root_heading: yes
    selection:
      members:
        - PINNtrain_IVP_Hard
        - train_network_IVP_Hard