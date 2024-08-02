from .SpecificTraining.IVP_Training import ode_TrainingIVP_soft as trainIVP_soft
from .SpecificTraining.IVP_Training import ode_TrainingIVP_Hard as trainIVP_Hard

from .SpecificTraining.BVP_Training import ode_TrainingBVP_soft as trainBVP_soft
from .SpecificTraining.BVP_Training import ode_TrainingBVP_Hard as trainBVP_Hard

from .SpecificTraining.IVP_Training import ode_Training2SystemIVP_soft as train2SystemIVP_soft
from .SpecificTraining.IVP_Training import ode_Training3SystemIVP_soft as train3SystemIVP_soft

from .SpecificTraining.DeepONet_Training.DeepONet_IVP import ode_TrainingDeepONetIVPsoft_order1 as DeepONet1IVPsoft
from .SpecificTraining.DeepONet_Training.DeepONet_IVP import ode_TrainingDeepONetIVPsoft_order2 as DeepONet2IVPsoft
from .SpecificTraining.DeepONet_Training.DeepONet_IVP import ode_TrainingDeepONetIVPsoft_order3 as DeepONet3IVPsoft
from .SpecificTraining.DeepONet_Training.DeepONet_IVP import ode_TrainingDeepONetIVPhard_order1 as DeepONet1IVPhard
from .SpecificTraining.DeepONet_Training.DeepONet_IVP import ode_TrainingDeepONetIVPhard_order2 as DeepONet2IVPhard
from .SpecificTraining.DeepONet_Training.DeepONet_IVP import ode_TrainingDeepONetIVPhard_order3 as DeepONet3IVPhard
from .SpecificTraining.DeepONet_Training.DeepONet_BVP import ode_TrainingDeepONetBVPhard_order12 as DeepONet12BVPhard
from .SpecificTraining.DeepONet_Training.DeepONet_BVP import ode_TrainingDeepONetBVPhard_order3 as DeepONet3BVPhard
from .SpecificTraining.DeepONet_Training.DeepONet_BVP import ode_TrainingDeepONetBVPsoft_order12 as DeepONet12BVPsoft
from .SpecificTraining.DeepONet_Training.DeepONet_BVP import ode_TrainingDeepONetBVPsoft_order3 as DeepONet3BVPsoft
from .SpecificTraining.HyperDeepONet_Training import ode_TrainingHyperDeepONetIVP_order1 as HyperDeepONet1IVP

from .SpecificTraining.DeepONet_Training.DeepONet_2EquationSystems import ode_TrainingDeepONet_2System_orders11 as DeepONetSys2_11
from .SpecificTraining.DeepONet_Training.DeepONet_2EquationSystems import ode_TrainingDeepONet_2System_orders21 as DeepONetSys2_21
from .SpecificTraining.DeepONet_Training.DeepONet_2EquationSystems import ode_TrainingDeepONet_2System_orders22 as DeepONetSys2_22
from .SpecificTraining.DeepONet_Training.DeepONet_2EquationSystems import ode_TrainingDeepONet_2System_orders31 as DeepONetSys2_31
from .SpecificTraining.DeepONet_Training.DeepONet_2EquationSystems import ode_TrainingDeepONet_2System_orders32 as DeepONetSys2_32
from .SpecificTraining.DeepONet_Training.DeepONet_2EquationSystems import ode_TrainingDeepONet_2System_orders33 as DeepONetSys2_33
from .SpecificTraining.DeepONet_Training.DeepONet_3EquationSystems import ode_TrainingDeepONet_3System_orders111 as DeepONetSys3_111
from .SpecificTraining.DeepONet_Training.DeepONet_3EquationSystems import ode_TrainingDeepONet_3System_orders211 as DeepONetSys3_211
from .SpecificTraining.DeepONet_Training.DeepONet_3EquationSystems import ode_TrainingDeepONet_3System_orders311 as DeepONetSys3_311
from .SpecificTraining.DeepONet_Training.DeepONet_3EquationSystems import ode_TrainingDeepONet_3System_orders221 as DeepONetSys3_221
from .SpecificTraining.DeepONet_Training.DeepONet_3EquationSystems import ode_TrainingDeepONet_3System_orders321 as DeepONetSys3_321
from .SpecificTraining.DeepONet_Training.DeepONet_3EquationSystems import ode_TrainingDeepONet_3System_orders331 as DeepONetSys3_331
from .SpecificTraining.DeepONet_Training.DeepONet_3EquationSystems import ode_TrainingDeepONet_3System_orders222 as DeepONetSys3_222
from .SpecificTraining.DeepONet_Training.DeepONet_3EquationSystems import ode_TrainingDeepONet_3System_orders322 as DeepONetSys3_322
from .SpecificTraining.DeepONet_Training.DeepONet_3EquationSystems import ode_TrainingDeepONet_3System_orders332 as DeepONetSys3_332
from .SpecificTraining.DeepONet_Training.DeepONet_3EquationSystems import ode_TrainingDeepONet_3System_orders333 as DeepONetSys3_333

def PINNtrainSelect_Standard(de_points, inits, t, epochs, eqn, order, net_layers, net_units, constraint, model, flag): 

   """
   Main selecting function which determines what problem is being solved and directs information from solveODE
   call and solution class to the correct training file

   Args:
      de_points (list): Randomly sampled points network uses to learn
      inits (list): Inital/Boundary values for each derivative
      t (list): Randomly sampled points along t
      epochs (int): Number of epochs model gets trained for
      eqn (string/list): Equation(s) to be solved
      order (int/list): Order(s) of equation(s)
      net_layers (int): Number of internal layers for network
      net_units (int): Number of nodes for each internal layer for network
      constraint (string): Constraint of inital conditions, "soft" or "hard'
      model (PINN): Model directly from solveODE call. If None, model will be created from an ode_ModelFuncs
         function. If not None, will try and use input model for training.
      flag (string): Internal flag for what type of equation is being solved

   Returns:
      epoch_loss (list): Total loss over training of model
      vp_loss (list): Inital Value or Boundary value loss over training of model
      de_loss (list): Differential equation loss over training of model
      model (PINN): Trained model to predict equation(s) over t
   """

   if flag == "IVP":
      if constraint == "soft":
         return trainIVP_soft.PINNtrain_IVP(de_points, inits, order, t[0], epochs, eqn, net_layers, net_units, model)
         
      elif constraint == "hard":
         return trainIVP_Hard.PINNtrain_IVP_Hard(de_points, inits, order, t, epochs, eqn, net_layers, net_units, model)
  
   elif flag == "BVP":
      if constraint == "soft":
         return trainBVP_soft.PINNtrain_BVP(de_points, inits, order, t, epochs, eqn, net_layers, net_units, model)
     
      elif constraint == "hard":
         return trainBVP_Hard.PINNtrain_BVP_Hard(de_points, inits, order, t, epochs, eqn, net_layers, net_units, model)
     
   elif flag == "System_IVP":
      if constraint == "soft":
         if len(order) == 2:
            return train2SystemIVP_soft.PINNtrain_2System_IVP(de_points, inits, order, t[0], epochs, eqn, net_layers, net_units, model)
         elif len(order) == 3:
            return train3SystemIVP_soft.PINNtrain_3System_IVP(de_points, inits, order, t[0], epochs, eqn, net_layers, net_units, model)
   else:
      raise Exception("Order/Constraint Not supported, please review")
         
def PINNtrainSelect_DeepONet(de_points, inits, t, epochs, eqn, order, orig_orders, N_sensors, sensor_range, net_layers, net_units, constraint, flag):
   """
   Main selecting function which determines what problem is being solved and directs information from solveODE_DeepONet
   call and solution class to the correct training file

   Args:
      de_points (list): Number of points to sample. Input N_pde.
      inits (list): Inital/Boundary values for each derivative
      t (list): Randomly sampled points along t
      epochs (int): Number of epochs model gets trained for
      eqn (string): Equation(s) to be solved
      order (int/list): Order(s) of equation(s)
      orig_orders (int/list): Original orders of equations input. Rearranged in corresponding solution class
      N_sensors (int): Number of sensors in which network learns over.
      sensor_range (list): range in which sensors are sampled over.
      net_layers (int): Number of internal layers for network
      net_units (int): Number of nodes for each internal layer for network
      constraint (string): Constraint of inital conditions, "soft" or "hard"
      flag (string): Internal flag for what type of equation is being solved

   Returns:
      loss (list): Total loss over training of model
      t (list): Evenly spaced points over t
      solPred (list): Solution prediction of trained model
      params (array): Trained parameters of model
      model (DeepONet): Trained model to predict equation(s) over t
      de_points (list): Randomly sampled points that model trains with
      sensors (list): Randomly sampled sensors the model trains with
   """

   if flag == "DeepONetIVP":
      if constraint == "hard":
         if order == 1:
            return DeepONet1IVPhard.startTraining(de_points, inits, order, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         elif order == 2:
            return DeepONet2IVPhard.startTraining(de_points, inits, order, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         elif order == 3:
            return DeepONet3IVPhard.startTraining(de_points, inits, order, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
      elif constraint == "soft":
         if order == 1:
            return DeepONet1IVPsoft.startTraining(de_points, inits, order, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         elif order == 2:
            return DeepONet2IVPsoft.startTraining(de_points, inits, order, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         elif order == 3:
            return DeepONet3IVPsoft.startTraining(de_points, inits, order, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)

   elif flag == "DeepONetBVP":
      if constraint == "hard":
         if order == 1 or order == 2:
            return DeepONet12BVPhard.startTraining(de_points, inits, order, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         elif order == 3:
            return DeepONet3BVPhard.startTraining(de_points, inits, order, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
      elif constraint == "soft":
         if order == 1 or order == 2:
            return DeepONet12BVPsoft.startTraining(de_points, inits, order, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         elif order == 3:
            return DeepONet3BVPsoft.startTraining(de_points, inits, order, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
      
   elif flag == "DeepONetSys":
      if len(order) == 2:
         if ((order[0] == 2) and (order[1] == 1)):
            return DeepONetSys2_21.startTraining(de_points, inits, orig_orders, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         elif ((order[0] == 1) and (order[1] == 1)):
            return DeepONetSys2_11.startTraining(de_points, inits, orig_orders, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         elif ((order[0] == 2) and (order[1] == 2)):
            return DeepONetSys2_22.startTraining(de_points, inits, orig_orders, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         elif ((order[0] == 3) and (order[1] == 1)):
            return DeepONetSys2_31.startTraining(de_points, inits, orig_orders, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         elif ((order[0] == 3) and (order[1] == 2)):
            return DeepONetSys2_32.startTraining(de_points, inits, orig_orders, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         elif ((order[0] == 3) and (order[1] == 3)):
            return DeepONetSys2_33.startTraining(de_points, inits, orig_orders, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)

      elif len(order) == 3:
         if ((order[0]==1) and (order[1]==1) and (order[2]==1)):
            return DeepONetSys3_111.startTraining(de_points, inits, orig_orders, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         elif ((order[0]==2) and (order[1]==1) and (order[2]==1)):
            return DeepONetSys3_211.startTraining(de_points, inits, orig_orders, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         elif ((order[0]==3) and (order[1]==1) and (order[2]==1)):
            return DeepONetSys3_311.startTraining(de_points, inits, orig_orders, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         elif ((order[0]==2) and (order[1]==2) and (order[2]==1)):
            return DeepONetSys3_221.startTraining(de_points, inits, orig_orders, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         elif ((order[0]==3) and (order[1]==2) and (order[2]==1)):
            return DeepONetSys3_321.startTraining(de_points, inits, orig_orders, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         elif ((order[0]==3) and (order[1]==3) and (order[2]==1)):
            return DeepONetSys3_331.startTraining(de_points, inits, orig_orders, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         elif ((order[0]==3) and (order[1]==3) and (order[2]==2)):
            return DeepONetSys3_332.startTraining(de_points, inits, orig_orders, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         elif ((order[0]==2) and (order[1]==2) and (order[2]==2)):
            return DeepONetSys3_222.startTraining(de_points, inits, orig_orders, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         elif ((order[0]==3) and (order[1]==2) and (order[2]==2)):
            return DeepONetSys3_322.startTraining(de_points, inits, orig_orders, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         elif ((order[0]==3) and (order[1]==3) and (order[2]==3)):
            return DeepONetSys3_333.startTraining(de_points, inits, orig_orders, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
         
   elif flag == "HyperDeepONetIVP":
      if order == 1:
         return HyperDeepONet1IVP.startTraining(de_points, inits, order, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn)
   
   else:
      raise Exception("Order/Constraint Not supported, please review")