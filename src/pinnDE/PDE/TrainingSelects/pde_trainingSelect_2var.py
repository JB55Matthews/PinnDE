from ..SpecificTraining.training_2variables.ICBCP_Training import pde_TrainingICBCP_periodic_soft as trainICBCP_periodic_soft
from ..SpecificTraining.training_2variables.BCP_Training import pde_TrainingBCP_dirichlet_soft as trainBCP_dirichlet_soft
from ..SpecificTraining.training_2variables.BCP_Training import pde_TrainingBCP_periodic as trainBCP_periodic
from ..SpecificTraining.training_2variables.ICBCP_Training import pde_TrainingICBCP_dirichlet_soft as trainICBCP_dirichlet_soft
from ..SpecificTraining.training_2variables.ICBCP_Training import pde_TrainingICBCP_hard as trainICBCP_hard
from ..SpecificTraining.training_2variables.BCP_Training import pde_TrainingBCP_dirichlet_hard as trainBCP_dirichlet_hard
from ..SpecificTraining.training_2variables.BCP_Training import pde_TrainingBCP_neumann_soft as trainBCP_neumann_soft
from ..SpecificTraining.training_2variables.ICBCP_Training import pde_TrainingICBCP_neumann_soft as trainICBCP_neumann_soft

from ..SpecificTraining.DeepONetTraining_2variables.DeepONet_ICBCP import pde_TrainingDeepONetICBCP_periodic_hard as trainICBCPDON_periodic_hard
from ..SpecificTraining.DeepONetTraining_2variables.DeepONet_ICBCP import pde_TrainingDeepONetICBCP_periodic_soft as trainICBCPDON_periodic_soft
from ..SpecificTraining.DeepONetTraining_2variables.DeepONet_ICBCP import pde_TrainingDeepONetICBCP_dirichlet_soft as trainICBCPDON_dirichlet_soft
from ..SpecificTraining.DeepONetTraining_2variables.DeepONet_ICBCP import pde_TrainingDeepONetICBCP_neumann_soft as trainICBCPDON_neumann_soft
from ..SpecificTraining.DeepONetTraining_2variables.DeepONet_BCP import pde_TrainingDeepONetBCP_hard as trainBCPDON_hard
from ..SpecificTraining.DeepONetTraining_2variables.DeepONet_BCP import pde_TrainingDeepONetBCP_dirichlet_soft as trainBCPDON_dirichlet_soft
from ..SpecificTraining.DeepONetTraining_2variables.DeepONet_BCP import pde_TrainingDeepONetBCP_neumann_soft as trainBCPDON_neumann_soft


def PINNtrainSelect_tx(pde_points, init_points, epochs, eqn, t_order, N_pde, N_iv, setup_boundries, model, 
                       constraint, extra_ders, flag): #selecting which to train
   """
   Main selecting function for solvePDE_tx which determines what problem is being solved and directs information
   to the correct training file

   Args:
      pde_points (list): pde_points returned from defineCollocationPoints_tx()
      init_points (list): inits returned from defineCollocationPoints_tx()
      epochs (int): Number of epochs model gets trained for
      eqn (string): Equation to be solved
      t_order (int): Order of t in equation 
      N_pde (int): Number of randomly sampled collocation points along t and x which PINN uses in training.
      N_iv (int): Number of randomly sampled collocation points along inital t which PINN uses in training.
      setup_boundries (boundary): boundary conditions set up from return of pde_Boundries_2var call.
      model (PINN): Model created from pde_ModelFuncs or input model
      constraint (string): Constraint of inital conditions, "soft" or "hard"
      extra_ders (list): Extra derivatives needed to be computed for user equation
      flag (string): Internal flag for what type of equation is being solved

   Returns:
      epoch_loss (list): Total loss over training of model
      iv_loss (list): Inital value loss over training of model
      bc_loss (list): Boundary condition loss over training of model
      pde_loss (list): Differential equation loss over training of model
      model (PINN): Trained model to predict equation solution
   """
   boundry_type = setup_boundries[0]

   if flag == "ICBCP-tx":
      if boundry_type == "periodic_timeDependent":
         if constraint == "soft":
            return trainICBCP_periodic_soft.PINNtrain(pde_points, init_points, t_order, setup_boundries, epochs, eqn, 
                                                      N_pde, N_iv, model, extra_ders)
         elif constraint == "hard":
               return trainICBCP_hard.PINNtrain(pde_points, init_points, t_order, setup_boundries, epochs, eqn, 
                                                N_pde, N_iv, model, extra_ders)
         
      elif boundry_type == "dirichlet_timeDependent":
         if constraint == "soft":
            return trainICBCP_dirichlet_soft.PINNtrain(pde_points, init_points, t_order, setup_boundries, epochs, eqn, 
                                                       N_pde, N_iv, model, extra_ders)
         elif constraint == "hard":
            return trainICBCP_hard.PINNtrain(pde_points, init_points, t_order, setup_boundries, epochs, eqn, 
                                             N_pde, N_iv, model, extra_ders)
         
      elif boundry_type == "neumann_timeDependent":
         if constraint == "soft":
            return trainICBCP_neumann_soft.PINNtrain(pde_points, init_points, t_order, setup_boundries, epochs, eqn, 
                                                     N_pde, N_iv, model, extra_ders)
      
   else:
      raise Exception("Order/Constraint Not supported, please review")
   
   

def PINNtrainSelect_xy(pde_points, epochs, eqn, N_pde, setup_boundries, model, constraint, extra_ders, flag): #selecting which to train
   """
   Main selecting function for solvePDE_xy which determines what problem is being solved and directs information
   to the correct training file

   Args:
      pde_points (list): pde_points returned from defineCollocationPoints_xy()
      epochs (int): Number of epochs model gets trained for
      eqn (string): Equation to be solved
      N_pde (int): Number of randomly sampled collocation points along x and y which PINN uses in training.
      setup_boundries (boundary): boundary conditions set up from return of pde_Boundries_2var call.
      model (PINN): Model created from pde_ModelFuncs or input model
      constraint (string): Constraint of inital conditions, "soft" or "hard"
      extra_ders (list): Extra derivatives needed to be computed for user equation
      flag (string): Internal flag for what type of equation is being solved

   Returns:
      epoch_loss (list): Total loss over training of model
      iv_loss (list): Inital value loss over training of model
      bc_loss (list): Boundary condition loss over training of model
      pde_loss (list): Differential equation loss over training of model
      model (PINN): Trained model to predict equation solution
   """
   boundry_type = setup_boundries[0]

   if flag == "BCP-xy":
      if boundry_type == "dirichlet_timeIndependent":
         if constraint == "soft":
            return trainBCP_dirichlet_soft.PINNtrain(pde_points, setup_boundries, epochs, eqn, N_pde, model, extra_ders)
         elif constraint == "hard":
            return trainBCP_dirichlet_hard.PINNtrain(pde_points, epochs, eqn, N_pde, model, extra_ders)
         
      elif boundry_type == "periodic_timeIndependent":
         if constraint == "soft":
            return trainBCP_periodic.PINNtrain(pde_points, setup_boundries, epochs, eqn, N_pde, model, extra_ders)
         elif constraint == "hard":
            return trainBCP_periodic.PINNtrain(pde_points, setup_boundries, epochs, eqn, N_pde, model, extra_ders)
         
      elif boundry_type == "neumann_timeIndependent":
         if constraint == "soft":
            return trainBCP_neumann_soft.PINNtrain(pde_points, setup_boundries, epochs, eqn, N_pde, model, extra_ders)
      
   else:
      raise Exception("Order/Constraint Not supported, please review")
   
def PINNtrainSelect_DeepONet_tx(pde_points, init_points, epochs, eqn, t_order, N_pde, N_iv, N_sensors, usensors, sensor_range, 
                                setup_boundries, model, constraint, extra_ders, flag): #selecting which to train
   """
   Main selecting function for solvePDE_DeepONet_tx which determines what problem is being solved and directs information
   to the correct training file

   Args:
      pde_points (list): pde_points returned from defineCollocationPoints_DON_tx()
      init_points (list): inits returned from defineCollocationPoints_DON_tx()
      epochs (int): Number of epochs model gets trained for
      eqn (string): Equation to be solved
      t_order (int): Order of t in equation 
      N_pde (int): Number of randomly sampled collocation points along t and x which DeepONet uses in training.
      N_iv (int): Number of randomly sampled collocation points along inital t which DeepONet uses in training.
      N_sensors (int): Number of sensors in which network learns over. 
      usensors (list): usensors returned from defineCollocationPoints_DON_tx()
      sensor_range (list): Range in which sensors are sampled over.
      setup_boundries (boundary): boundary conditions set up from return of pde_Boundries_2var call.
      model (DeepONet): Model created from pde_ModelFuncs or input model
      constraint (string): Constraint of inital conditions, "soft" or "hard"
      extra_ders (list): Extra derivatives needed to be computed for user equation
      flag (string): Internal flag for what type of equation is being solved

   Returns:
      epoch_loss (list): Total loss over training of model
      iv_loss (list): Inital value loss over training of model
      bc_loss (list): Boundary condition loss over training of model
      pde_loss (list): Differential equation loss over training of model
      model (DeepONet): Trained model to predict equation solution
   """
   
   boundry_type = setup_boundries[0]

   if flag == "DeepONet-ICBCP-tx":
      if boundry_type == "periodic_timeDependent":
         if constraint == "soft":
            return trainICBCPDON_periodic_soft.train(pde_points, init_points, usensors, t_order, epochs, model, eqn, N_iv, extra_ders)
         elif constraint == "hard":
               return trainICBCPDON_periodic_hard.train(pde_points, usensors, epochs, model, eqn, extra_ders)
         
      elif boundry_type == "dirichlet_timeDependent":
         if constraint == "soft":
            return trainICBCPDON_dirichlet_soft.train(pde_points, init_points, usensors, setup_boundries, t_order, epochs, 
                                                      model, eqn, N_iv, extra_ders)
         elif constraint == "hard":
            return #trainIBCP_hard.PINNtrain(pde_points, init_points, t_order, setup_boundries, epochs, eqn, N_pde, N_iv, model)
         
      elif boundry_type == "neumann_timeDependent":
         if constraint == "soft":
            return trainICBCPDON_neumann_soft.train(pde_points, init_points, usensors, setup_boundries, t_order, epochs, 
                                                    model, eqn, N_iv, extra_ders)
      
   else:
      raise Exception("Order/Constraint Not supported, please review")


def PINNtrainSelect_DeepONet_xy(pde_points, epochs, eqn, N_pde, N_bc, N_sensors, usensors, sensor_range, setup_boundries, 
                                model, constraint, extra_ders, flag): #selecting which to train
   """
   Main selecting function for solvePDE_DeepONet_xy which determines what problem is being solved and directs information
   to the correct training file

   Args:
      pde_points (list): pde_points returned from defineCollocationPoints_DON_xy()
      epochs (int): Number of epochs model gets trained for
      eqn (string): Equation to be solved
      N_pde (int): Number of randomly sampled collocation points along x and y which DeepONet uses in training.
      N_bc (int): Number of randomly sampled collocation points along boundary which DeepONet uses in training.
      N_sensors (int): Number of sensors in which network learns over. 
      usensors (list): usensors returned from defineCollocationPoints_DON_tx()
      sensor_range (list): Range in which sensors are sampled over.
      setup_boundries (boundary): boundary conditions set up from return of pde_Boundries_2var call.
      model (DeepONet): Model created from pde_ModelFuncs or input model
      constraint (string): Constraint of inital conditions, "soft" or "hard"
      extra_ders (list): Extra derivatives needed to be computed for user equation
      flag (string): Internal flag for what type of equation is being solved

   Returns:
      epoch_loss (list): Total loss over training of model
      iv_loss (list): Inital value loss over training of model
      bc_loss (list): Boundary condition loss over training of model
      pde_loss (list): Differential equation loss over training of model
      model (DeepONet): Trained model to predict equation solution
   """
   boundry_type = setup_boundries[0]

   if flag == "DeepONet-BCP-xy":
      if boundry_type == "dirichlet_timeIndependent":
         if constraint == "soft":
            return trainBCPDON_dirichlet_soft.train(pde_points, usensors, setup_boundries, epochs, model, eqn, N_bc, extra_ders)
         elif constraint == "hard":
            return trainBCPDON_hard.train(pde_points, usensors, epochs, model, eqn, N_bc, extra_ders)
         
      elif boundry_type == "periodic_timeIndependent":
         if constraint == "soft":
            return trainBCPDON_hard.train(pde_points, usensors, epochs, model, eqn, N_bc, extra_ders)
         elif constraint == "hard":
            return trainBCPDON_hard.train(pde_points, usensors, epochs, model, eqn, N_bc, extra_ders)
         
      elif boundry_type == "neumann_timeIndependent":
         if constraint == "soft":
            return trainBCPDON_neumann_soft.train(pde_points, usensors, setup_boundries, epochs, model, eqn, N_bc, extra_ders)
      
   else:
      raise Exception("Order/Constraint Not supported, please review")
   
        