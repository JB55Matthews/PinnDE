# pde_TrainingDeepONetBCP_dirichlet_soft

Functions which trains a model for solving a solvePDE_DeepONet_xy call with dirichlet boundries and soft constraints implemented in TensorFlow.

While available to user, not meant to be used. Meant to be used through
object returned from solvePDE calls, where training file is selected through pde_trainingSelects.PINNtrainSelect_DeepONet_xy()

::: src.pinnDE.PDE.SpecificTraining.DeepONetTraining_2variables.DeepONet_BCP.pde_TrainingDeepONetBCP_dirichlet_soft
    rendering:
      show_root_heading: yes
    selection:
      members:
        - train
        - train_network