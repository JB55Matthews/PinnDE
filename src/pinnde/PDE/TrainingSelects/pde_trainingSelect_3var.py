# import SpecificTraining.training_2variables.BCP_Training.pde_TrainingBCP_dirichlet_soft as trainBCP_dirichlet_soft
# import SpecificTraining.training_2variables.BCP_Training.pde_TrainingBCP_periodic as trainBCP_periodic
# import SpecificTraining.training_2variables.IBCP_Training.pde_TrainingIBCP_dirichlet_soft as trainIBCP_dirichlet_soft
# import SpecificTraining.training_2variables.IBCP_Training.pde_TrainingIBCP_hard as trainIBCP_hard
# import SpecificTraining.training_2variables.BCP_Training.pde_TrainingBCP_dirichlet_hard as trainBCP_dirichlet_hard
# import SpecificTraining.training_2variables.BCP_Training.pde_TrainingBCP_neumann_soft as trainBCP_neumann_soft
# import SpecificTraining.training_2variables.IBCP_Training.pde_TrainingIBCP_neumann_soft as trainIBCP_neumann_soft

# import SpecificTraining.DeepONetTraining_2variables.DeepONet_IBCP.pde_TrainingDeepONetIBCP_periodic_hard as trainIBCPDON_periodic_hard
# import SpecificTraining.DeepONetTraining_2variables.DeepONet_IBCP.pde_TrainingDeepONetIBCP_periodic_soft as trainIBCPDON_periodic_soft
# import SpecificTraining.DeepONetTraining_2variables.DeepONet_IBCP.pde_TrainingDeepONetIBCP_dirichlet_soft as trainIBCPDON_dirichlet_soft
# import SpecificTraining.DeepONetTraining_2variables.DeepONet_IBCP.pde_TrainingDeepONetIBCP_neumann_soft as trainIBCPDON_neumann_soft
# import SpecificTraining.DeepONetTraining_2variables.DeepONet_BCP.pde_TrainingDeepONetBCP_periodic_hard as trainBCPDON_periodic_hard
# import SpecificTraining.DeepONetTraining_2variables.DeepONet_BCP.pde_TrainingDeepONetBCP_dirichlet_soft as trainBCPDON_dirichlet_soft
# import SpecificTraining.DeepONetTraining_2variables.DeepONet_BCP.pde_TrainingDeepONetBCP_neumann_soft as trainBCPDON_neumann_soft

from ..SpecificTraining.training_3variables.ICBCP3_Training import pde_TrainingICBCP3_periodic_soft as trainICBCP3_periodic_soft
from ..SpecificTraining.training_3variables.ICBCP3_Training import pde_TrainingICBCP3_periodic_hard as trainICBCP3_periodic_hard
from ..SpecificTraining.training_3variables.ICBCP3_Training import pde_TrainingICBCP3_dirichlet_soft as trainICBCP3_dirichlet_soft



def PINNtrainSelect_txy(pde_points, init_points, epochs, eqn, t_order, N_pde, N_iv, setup_boundaries, model, constraint, flag): #selecting which to train
   boundary_type = setup_boundaries[0]

   if flag == "ICBCP-txy":
      if boundary_type == "periodic_timeDependent":
         if constraint == "soft":
            return trainICBCP3_periodic_soft.PINNtrain(pde_points, init_points, t_order, setup_boundaries, epochs, eqn, N_pde, N_iv, model)
         elif constraint == "hard":
               return trainICBCP3_periodic_hard.PINNtrain(pde_points, setup_boundaries, epochs, eqn, N_pde, N_iv, model)
         
      elif boundary_type == "dirichlet_timeDependent":
         if constraint == "soft":
            return trainICBCP3_dirichlet_soft.PINNtrain(pde_points, init_points, t_order, setup_boundaries, epochs, eqn, N_pde, N_iv, model)
         elif constraint == "hard":
            return #trainIBCP_hard.PINNtrain(pde_points, init_points, t_order, setup_boundaries, epochs, eqn, N_pde, N_iv, model)
         
      elif boundary_type == "neumann_timeDependent":
         if constraint == "soft":
            return #trainIBCP_neumann_soft.PINNtrain(pde_points, init_points, t_order, setup_boundaries, epochs, eqn, N_pde, N_iv, model)
      
   else:
      raise Exception("Order/Constraint Not supported, please review")
   
   
        