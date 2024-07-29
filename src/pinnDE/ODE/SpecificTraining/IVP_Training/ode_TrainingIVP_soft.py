import numpy as np
import tensorflow as tf
from ...ode_ModelFuncs import build_model
import ast

#The main trainign function
@tf.function
def train_network_general_IVP(odes, inits, order, model, gamma, eqnparam):
    """
    Function which does the training for a single epoch

    Args:
      odes (list): Sampled de points network uses to train
      inits (list): Inital value points for learning initial value
      order (int): Order of equation to solve
      model (PINN): Model to train
      gamma (float): Weight of IV loss when added with DE loss
      eqnparam (string): Equation to solve.

    Generates derivatives of model using automatic differentiation. Computes
    mean squared error of loss along odes points and for inital values.

    Returns:
      DEloss (list): Loss of training network to match function along odes points
      IVloss (list): Loss of training network to match initial values
      grads (array): Gradients of network for optimization
    """

    # DE collocation points
    t_de = odes[:,:1]
    
    # Initial value points
    if order == 1:
      t_init, u_init = inits[:1], inits[1:2]
    elif order == 2:
      t_init, u_init, ux_init = inits[:1], inits[1:2], inits[2:3]
    elif order == 3:
      t_init, u_init, ux_init, uxx_init = inits[:1], inits[1:2], inits[2:3], inits[3:4]
    elif order == 4:
      t_init, u_init, ux_init, uxx_init = inits[:1], inits[1:2], inits[2:3], inits[3:4]
      uxxx_init = inits[4:5]
    else:
      t_init, u_init, ux_init, uxx_init = inits[:1], inits[1:2], inits[2:3], inits[3:4]
      uxxx_init, uxxxx_init = inits[4:5], inits[5:6]

    # Outer gradient for tuning network parameters
    with tf.GradientTape() as tape:

      # Inner gradient for derivatives of u wrt x and t
      with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t_de)
        u = model(t_de)
        ut = tape2.gradient(u, t_de)
        utt = tape2.gradient(ut, t_de)
        uttt = tape2.gradient(utt, t_de)
        utttt = tape2.gradient(uttt, t_de)
        uttttt = tape2.gradient(utttt, t_de)
      
      t = t_de

      # Define the differential equation loss
      parse_tree = ast.parse(eqnparam, mode="eval")
      eqn = eval(compile(parse_tree, "<string>", "eval"))

      DEloss = tf.reduce_mean(tf.square(eqn))

      # Define the initial value loss
      with tf.GradientTape(persistent=True) as tape3:
        tape3.watch(t_init)
        u_init_pred = model(t_init)
        if order > 1:
          ux_init_pred = tape3.gradient(u_init_pred, t_init)
          if order > 2:
            uxx_init_pred = tape3.gradient(ux_init_pred, t_init)
            if order > 3:
              uxxx_init_pred = tape3.gradient(uxx_init_pred, t_init)
              if order > 4:
                uxxxx_init_pred = tape3.gradient(uxxx_init_pred, t_init)

      if order == 1:
        IVloss = tf.reduce_mean(tf.square(u_init_pred - u_init))
      elif order == 2:
        IVloss = tf.reduce_mean(tf.square(u_init_pred - u_init) + tf.square(ux_init_pred - ux_init))
      elif order == 3:
        IVloss = tf.reduce_mean(tf.square(u_init_pred - u_init) + tf.square(ux_init_pred - ux_init) +
                               tf.square(uxx_init_pred - uxx_init))
      elif order == 4:
        IVloss = tf.reduce_mean(tf.square(u_init_pred - u_init) + tf.square(ux_init_pred - ux_init) +
                               tf.square(uxx_init_pred - uxx_init) + tf.square(uxxx_init_pred - uxxx_init))
      else:
        IVloss = tf.reduce_mean(tf.square(u_init_pred - u_init) + tf.square(ux_init_pred - ux_init) +
                               tf.square(uxx_init_pred - uxx_init) + tf.square(uxxx_init_pred - uxxx_init) +
                               tf.square(uxxxx_init_pred - uxxxx_init))
      

      # Composite loss function
      loss = DEloss + gamma*IVloss

    grads = tape.gradient(loss, model.trainable_variables)
    return DEloss, IVloss, grads


def PINNtrain_IVP(de_points, inits, order, t0, epochs, eqn, net_layers, net_units, model):
  """
  Main function called by PINNtrainSelect_Standard when solving soft constraint IVP.

  Args:
    de_points (list): Randomly sampled points for network to train with.
    inits (list): Inital values for network to learn
    order (int): Order of equation to be solved
    t0 (float): First value in user input t_bdry
    epochs (int): Number of epochs for network to train
    eqn (string): Equation to solve. User input eqn
    net_layers (int): Number of internal layers of network
    net_units (int): Number of nodes for each internal layer
    model (PINN or None): User input model. Defaulted to None and model created in function
      with call to ode_ModelFuncs

  Returns:
    epoch_loss (list): Total loss over training of model
    ivp_loss (list): Inital Value loss over training of model
    de_loss (list): Differential equation loss over training of model
    model (PINN): Trained model to predict equation(s) over t

  Packages data correctly and calls train_network_general_IVP in executing training routine, and handles
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

  # Initial value
  if order == 1:
    t_init, u_init = np.array([t0]).astype(np.float32), np.array([inits[0]]).astype(np.float32)
  elif order == 2:
    t_init, u_init = np.array([t0]).astype(np.float32), np.array([inits[0]]).astype(np.float32)
    ux_init = np.array([inits[1]]).astype(np.float32)
  elif order == 3:
    t_init, u_init = np.array([t0]).astype(np.float32), np.array([inits[0]]).astype(np.float32)
    ux_init, uxx_init = np.array([inits[1]]).astype(np.float32), np.array([inits[2]]).astype(np.float32)
  elif order == 4:
    t_init, u_init = np.array([t0]).astype(np.float32), np.array([inits[0]]).astype(np.float32)
    ux_init, uxx_init = np.array([inits[1]]).astype(np.float32), np.array([inits[2]]).astype(np.float32)
    uxxx_init = np.array(inits[3]).astype(np.float32)
  else:
    t_init, u_init = np.array([t0]).astype(np.float32), np.array([inits[0]]).astype(np.float32)
    ux_init, uxx_init = np.array([inits[1]]).astype(np.float32), np.array([inits[2]]).astype(np.float32)
    uxxx_init, uxxxx_init = np.array(inits[3]).astype(np.float32), np.array(inits[4]).astype(np.float32)

  epoch_loss = np.zeros(epochs)
  ivp_loss = np.zeros(epochs)
  de_loss = np.zeros(epochs)
  nr_batches = 0

  # Generate the tf.Dataset for the initial collocation points
  if order == 1:
    inits = np.column_stack([t_init, u_init])
  elif order == 2:
    inits = np.column_stack([t_init, u_init, ux_init])
  elif order == 3:
    inits = np.column_stack([t_init, u_init, ux_init, uxx_init])
  elif order == 4:
    inits = np.column_stack([t_init, u_init, ux_init, uxx_init, uxxx_init])
  else:
    inits = np.column_stack([t_init, u_init, ux_init, uxx_init, uxxx_init, uxxxx_init])

  ds_init = tf.data.Dataset.from_tensor_slices(inits)
  ds_init = ds_init.cache()

  # Generate the tf.Dataset for the differential equations points
  ds_ode = tf.data.Dataset.from_tensor_slices(de_points.astype(np.float32))
  ds_ode = ds_ode.cache().shuffle(N_de).batch(bs_de)

  # Generate entire dataset
  ds = tf.data.Dataset.zip((ds_ode, ds_init))
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  # Generate the model
  opt = tf.keras.optimizers.Adam(lr_model)
  if model == None:
    model = build_model(net_layers, net_units)

  # Main training loop
  for i in range(epochs):

    # Training for that epoch
    for (des, inits) in ds:

      nr_batches = 0

      # Train the network
      DEloss, IVloss, grads = train_network_general_IVP(des, inits, order, model, gamma, eqn)

      # Gradient step
      opt.apply_gradients(zip(grads, model.trainable_variables))

      epoch_loss[i] += DEloss + gamma*IVloss
      ivp_loss[i] += IVloss
      de_loss[i] += DEloss
      nr_batches += 1

    # Get total epoch loss
    epoch_loss[i] /= nr_batches

    if (np.mod(i, 100)==0):
      print("DE loss, IV loss in {}th epoch: {: 6.4f}, {: 6.4f}.".format(i, DEloss, IVloss))

  return epoch_loss, ivp_loss, de_loss, model