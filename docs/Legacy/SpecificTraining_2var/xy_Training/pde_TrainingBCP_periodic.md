# pde_TrainingBCP_periodic

Functions which trains a model for solving a solvePDE_xy call with periodic boundaries implemented in TensorFlow.

While available to user, not meant to be used. Meant to be used through
object returned from solvePDE calls, where training file is selected through pde_trainingSelects.PINNtrainSelect_xy()

::: src.pinnde.legacy.PDE.SpecificTraining.training_2variables.BCP_Training.pde_TrainingBCP_periodic
    rendering:
      show_root_heading: yes
    selection:
      members:
        - PINNtrain
        - trainStep