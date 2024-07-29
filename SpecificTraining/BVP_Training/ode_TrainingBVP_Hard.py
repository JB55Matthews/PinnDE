import numpy as np
import tensorflow as tf
from ...ode_ModelFuncs import build_model_hardConstraint_order12_BVP
from ...ode_ModelFuncs import build_model_hardConstraint_order3_BVP
import ast

#The main trainign function
@tf.function
def train_network_BVP_Hard(odes, model, eqnparam):
    """
    Function which does the training for a single epoch

    Args:
      odes (list): Sampled de points network uses to train
      model (PINN): Model to train
      eqnparam (string): Equation to solve.

    Generates derivatives of model using automatic differentiation. Computes
    mean squared error of loss along odes points.

    Returns:
      DEloss (list): Loss of training network to match function along odes points
      grads (array): Gradients of network for optimization
    """

    # DE collocation points
    t_de = odes[:,:1]

    # Outer gradient for tuning network parameters
    with tf.GradientTape() as tape:

      # Inner gradient for derivatives of u wrt x and t
      with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t_de)
        u = model(t_de)
        ut = tape2.gradient(u, t_de)
        utt = tape2.gradient(ut, t_de)
        uttt = tape2.gradient(utt, t_de)
      t= t_de
      
      parse_tree = ast.parse(eqnparam, mode="eval")
      eqn = eval(compile(parse_tree, "<string>", "eval"))
            
      DEloss = tf.reduce_mean(tf.square(eqn))


    grads = tape.gradient(DEloss, model.trainable_variables)
    return DEloss, grads


def PINNtrain_BVP_Hard(de_points, inits, order, t_bdry, epochs, eqn, net_layers, net_units, model):
  """
  Main function called by PINNtrainSelect_Standard when solving hard constraint BVP.

  Args:
    de_points (list): Randomly sampled points for network to train with.
    inits (list): Inital boundary values for network to learn
    order (int): Order of equation to be solved
    t_bdry (list): Interval for equation to be solved on. User input t_bdry
    epochs (int): Number of epochs for network to train
    eqn (string): Equation to solve. User input eqn
    net_layers (int): Number of internal layers of network
    net_units (int): Number of nodes for each internal layer
    model (PINN or None): User input model. Defaulted to None and model created in function
      with call to ode_ModelFuncs

  Returns:
    epoch_loss (list): Total loss over training of model
    ivp_loss (list): Inital boundary value loss over training of model. Will be all zeroes as hard constrainting
    de_loss (list): Differential equation loss over training of model
    model (PINN): Trained model to predict equation(s) over t

  Packages data correctly and calls train_network_general_BVP_Hard in executing training routine, and handles
  optimization of the network.
  """

  # Total number of collocation points
  N_de = len(de_points)

  # Batch size
  bs_de = N_de

  # Weight factor gamma
  gamma = 1.0

  # Learning rate
  lr_model = 1e-3

  epoch_loss = np.zeros(epochs)
  ivp_loss = np.zeros(epochs)
  de_loss = np.zeros(epochs)
  nr_batches = 0

  # Generate the tf.Dataset for the differential equations points
  ds = tf.data.Dataset.from_tensor_slices(de_points.astype(np.float32))
  ds = ds.cache().shuffle(N_de).batch(bs_de)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  # Generate the model
  opt = tf.keras.optimizers.Adam(lr_model)

  #fix model
  if model == None:
    if ((order == 1) or (order == 2)):
      model_params = [t_bdry[0], t_bdry[1], inits[0], inits[1]]
      model = build_model_hardConstraint_order12_BVP(model_params, net_layers, net_units)
    elif order == 3:
      model_params = [t_bdry[0], t_bdry[1], inits[0], inits[1], inits[2], inits[3]]
      model = build_model_hardConstraint_order3_BVP(model_params, net_layers, net_units)

  # Main training loop
  for i in range(epochs):

    # Training for that epoch
    for des in ds:

      nr_batches = 0

      # Train the network
      DEloss, grads = train_network_BVP_Hard(des, model, eqn)

      # Gradient step
      opt.apply_gradients(zip(grads, model.trainable_variables))

      epoch_loss[i] += DEloss
      de_loss[i] += DEloss
      nr_batches += 1

    # Get total epoch loss
    epoch_loss[i] /= nr_batches

    if (np.mod(i, 100)==0):
      print("DE loss, IV loss in {}th epoch:{: 6.4f}.".format(i, DEloss))

  return epoch_loss, ivp_loss, de_loss, model