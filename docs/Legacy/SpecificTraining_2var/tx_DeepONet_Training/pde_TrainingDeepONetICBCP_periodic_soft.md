# pde_TrainingDeepONetICBCP_periodic_soft

Functions which trains a model for solving a solvePDE_DeepONet_tx call with periodic boundaries and soft constraint implemented in TensorFlow.

While available to user, not meant to be used. Meant to be used through
object returned from solvePDE calls, where training file is selected through pde_trainingSelects.PINNtrainSelect_DeepONet_tx()

::: src.pinnde.legacy.PDE.SpecificTraining.DeepONetTraining_2variables.DeepONet_ICBCP.pde_TrainingDeepONetICBCP_periodic_soft
    rendering:
      show_root_heading: yes
    selection:
      members:
        - train
        - train_network