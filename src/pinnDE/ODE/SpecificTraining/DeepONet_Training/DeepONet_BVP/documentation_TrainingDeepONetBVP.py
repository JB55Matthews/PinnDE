from flax import linen as nn

#--------------------------------------------------
# FILE FOR DOCUMENTING CODE THROUHGOUT THIS DIRECTORY
#-----------------------------------------------------

# Define the normalization layer
class Normalize(nn.Module):
  """
  Class which describes a normalize layer for DeepONet. Returns input data
  normalized to interval [-1, 1].
  """

  @nn.compact
  def __call__(self, x):
    return
  
class CombineBranches(nn.Module):
  """
  Class which combines data from two branch nets and returns resulting combination.
  """
  def __call__(self):
    return 
  
class HardConstraint(nn.Module):
  """
  Class which applys hard constraint of boundary values to network. Returns input data
  after being hard constrainted.
  """
  def __call__(self):
    return
  
class MLP(nn.Module):
  """
  Class which describes MLP used as basis for branch and trunk nets. Uses user
  input net_layers and net_units.
  """

  def __call__(self):
    return
  
class DeepONet(nn.Module):
  """
  Class which describes DeepONet model. Creates MLP-based trunk and branch networks for
  what is needed based on order of problem, normalizes, and hard constraints if hard constraint equation
  """

  def __call__(self):
    return
  
# Define the collocation points over the domain [t_0, t_1]
def defineCollocationPoints(t_bdry, N, sensor_range):
  """
  File specific collocationPoints function to generate sampled ode_points and sensors.
  Differnet orders require slightly different functions.

  Args:
    t_bdry (list): Interval of t to train on
    N (int): Number of sensor points
    sensor_range (list): Range to sample sensors over
  """
  return

# DeepONet training function (standard DeepONet)
def train_step(params, pdes, z):
  """
  Main training function. Defines derivatives of network, and defines loss function to
  minimzie input equation and inital values if soft constrainting.

  Args:
    params (array): paramters of DeepONet
    pdes (list): sampled de points to train with
    z (list): sensors for function u

  May also have arguments zt, ztt, etc. for derivative sensors
  """
  return


def train_network(params, des, zsensor, epochs):
  """
  Main function which calls train_step. Packages data, performs training routine,
  and does network optimization.

  Args:
    params (array): paramters of DeepONet
    des (list): Randomy sampled de points to train with
    zsensor (list): Randomly sampled sensors to train with
    epochs (int): Number of epochs to train network for
  """

  return

def startTraining(eval_points, inits, order, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn):
  """
  Main function of training, called by PINNtrainSelect_DeepONet.

  Args:
    eval_points (int): Number of points to sample along t. User innput N_pde.
    inits (list): Inital values. User input init.
    order (int): Order of equation. User input order.
    t (list): User input t_bdry
    N_sensors (int): Number of sensors to sample. User input N_sensors
    sensor_range (list): Range to take sensor rample over. User input sensor_range
    epochs (int): Number of epochs to train network for. User input epochs
    net_layers (int): Number of internal layers for network. User input net_layers
    net_units (int): Number of nodes for each internal layer. User input net_units
    eqn (string): Equation to solve. User input eqn.

  Function generates DeepONet, de_points, sensors, and parameters. Then calls train_network on DeepONet. Then gets 
  network solution prediction.

  Returns:
      loss (list): Total loss over training of model
      t (list): Evenly spaced points over t
      solPred (list): Solution prediction of trained model
      params (array): Trained parameters of model
      model (DeepONet): Trained model to predict equation(s) over t
      de_points (list): Randomly sampled points that model trains with
      sensors (list): Randomly sampled sensors the model trains with

  """
  loss = []
  t = []
  solPred = []
  params = []
  deeponet = []
  de_points  = []
  zsensors = []

  return loss, t, solPred, params, deeponet, de_points, zsensors