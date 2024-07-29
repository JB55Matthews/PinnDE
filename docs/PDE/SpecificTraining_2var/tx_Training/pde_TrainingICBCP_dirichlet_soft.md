# pde_TrainingICBCP_dirichlet_soft

Functions which trains a model for solving a solvePDE_tx call with dirichlet boundries and soft constraints implemented in TensorFlow.

While available to user, not meant to be used. Meant to be used through
object returned from solvePDE calls, where training file is selected through pde_trainingSelects.PINNtrainSelect_tx()

::: src.pinnDE.PDE.SpecificTraining.training_2variables.ICBCP_Training.pde_TrainingICBCP_dirichlet_soft
    rendering:
      show_root_heading: yes
    selection:
      members:
        - PINNtrain
        - trainStep