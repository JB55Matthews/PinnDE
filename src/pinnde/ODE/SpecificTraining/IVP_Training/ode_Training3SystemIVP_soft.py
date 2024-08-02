import numpy as np
import tensorflow as tf
from ...ode_ModelFuncs import build_model_system3_IVP
import ast

#The main trainign function
@tf.function
def train_network_3system_IVP(odes, inits, order, model, gamma, eqnparam):
    """
    Function which does the training for a single epoch

    Args:
      odes (list): Sampled de points network uses to train
      inits (list): Inital value points for learning initial value
      order (list): Orders of equations to solve
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
    if order[0] == 1:
      t_init, u_init = inits[:1], inits[1:2]
      if order[1] == 1:
        x_init = inits[2:3]
        if order[2] == 1:
          y_init = inits[3:4]
        elif order[2] == 2:
          y_init, yt_init = inits[3:4], inits[4:5]
        elif order[2] == 3:
          y_init, yt_init, ytt_init = inits[3:4], inits[4:5], inits[5:6]
      elif order[1] == 2:
        x_init, xt_init = inits[2:3], inits[3:4]
        if order[2] == 1:
          y_init = inits[4:5]
        elif order[2] == 2:
          y_init, yt_init = inits[4:5], inits[5:6]
        elif order[2] == 3:
          y_init, yt_init, ytt_init = inits[4:5], inits[5:6], inits[6:7]
      elif order[1] == 3:
        x_init, xt_init, xtt_init = inits[2:3], inits[3:4], inits[4:5]
        if order[2] == 1:
          y_init = inits[5:6]
        elif order[2] == 2:
          y_init, yt_init = inits[5:6], inits[6:7]
        elif order[2] == 3:
          y_init, yt_init, ytt_init = inits[5:6], inits[6:7], inits[7:8]
    elif order[0] == 2:
      t_init, u_init, ut_init = inits[:1], inits[1:2], inits[2:3]
      if order[1] == 1:
        x_init = inits[3:4]
        if order[2] == 1:
          y_init = inits[4:5]
        elif order[2] == 2:
          y_init, yt_init = inits[4:5], inits[5:6]
        elif order[2] == 3:
          y_init, yt_init, ytt_init = inits[4:5], inits[5:6], inits[6:7]
      elif order[1] == 2:
        x_init, xt_init = inits[3:4], inits[4:5]
        if order[2] == 1:
          y_init = inits[5:6]
        elif order[2] == 2:
          y_init, yt_init = inits[5:6], inits[6:7]
        elif order[2] == 3:
          y_init, yt_init, ytt_init = inits[5:6], inits[6:7], inits[7:8]
      elif order[1] == 3:
        x_init, xt_init, xtt_init = inits[3:4], inits[4:5], inits[5:6]
        if order[2] == 1:
          y_init = inits[6:7]
        elif order[2] == 2:
          y_init, yt_init = inits[6:7], inits[7:8]
        elif order[2] == 3:
          y_init, yt_init, ytt_init = inits[6:7], inits[7:8], inits[8:9]
    elif order[0] == 3:
      t_init, u_init, ut_init, utt_init = inits[:1], inits[1:2], inits[2:3], inits[3:4]
      if order[1] == 1:
        x_init = inits[4:5]
        if order[2] == 1:
          y_init = inits[5:6]
        elif order[2] == 2:
          y_init, yt_init = inits[5:6], inits[6:7]
        elif order[2] == 3:
          y_init, yt_init, ytt_init = inits[5:6], inits[6:7], inits[7:8]
      elif order[1] == 2:
        x_init, xt_init = inits[4:5], inits[5:6]
        if order[2] == 1:
          y_init = inits[6:7]
        elif order[2] == 2:
          y_init, yt_init = inits[6:7], inits[7:8]
        elif order[2] == 3:
          y_init, yt_init, ytt_init = inits[6:7], inits[7:8], inits[8:9]
      elif order[1] == 3:
        x_init, xt_init, xtt_init = inits[4:5], inits[5:6], inits[6:7]
        if order[2] == 1:
          y_init = inits[7:8]
        elif order[2] == 2:
          y_init, yt_init = inits[7:8], inits[8:9]
        elif order[2] == 3:
          y_init, yt_init, ytt_init = inits[7:8], inits[8:9], inits[9:10]
      
      

    # Outer gradient for tuning network parameters
    with tf.GradientTape() as tape:

      # Inner gradient for derivatives of u wrt x and t
      with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t_de)
        [u, x, y] = model(t_de)
        ut = tape2.gradient(u, t_de)
        utt = tape2.gradient(ut, t_de)
        uttt = tape2.gradient(utt, t_de)
        xt = tape2.gradient(x, t_de)
        xtt = tape2.gradient(xt, t_de)
        xttt = tape2.gradient(xtt, t_de)
        yt = tape2.gradient(y, t_de)
        ytt = tape2.gradient(yt, t_de)
        yttt = tape2.gradient(ytt, t_de)
      
      t = t_de

      # Define the differential equation loss
      eqn1 = eqnparam[0]
      eqn2 = eqnparam[1]
      eqn3 = eqnparam[2]

      parse_tree = ast.parse(eqn1, mode="eval")
      eqn1parse = eval(compile(parse_tree, "<string>", "eval"))

      parse_tree2 = ast.parse(eqn2, mode="eval")
      eqn2parse = eval(compile(parse_tree2, "<string>", "eval"))

      parse_tree3 = ast.parse(eqn3, mode="eval")
      eqn3parse = eval(compile(parse_tree3, "<string>", "eval"))

      DEloss = tf.reduce_mean(tf.square(eqn1parse)) + tf.reduce_mean(tf.square(eqn2parse)) + tf.reduce_mean(tf.square(eqn3parse))

      # Define the initial value loss
      with tf.GradientTape(persistent=True) as tape3:
        tape3.watch(t_init)
        [u_init_pred, x_init_pred, y_init_pred] = model(t_init)
        if order[0] > 1:
          ut_init_pred = tape3.gradient(u_init_pred, t_init)
          if order[0] > 2:
            utt_init_pred = tape3.gradient(ut_init_pred, t_init)
        if order[1] > 1:
          xt_init_pred = tape3.gradient(x_init_pred, t_init)
          if order[1] > 2:
            xtt_init_pred = tape3.gradient(xt_init_pred, t_init)
        if order[2] > 1:
          yt_init_pred = tape3.gradient(y_init_pred, t_init)
          if order[2] > 2:
            ytt_init_pred = tape3.gradient(yt_init_pred, t_init)

        
      IVloss = 0
      if order[0] == 1:
        IVloss = IVloss + tf.reduce_mean(tf.square(u_init_pred - u_init))
      elif order[0] == 2:
        IVloss = IVloss + tf.reduce_mean(tf.square(u_init_pred - u_init) + tf.square(ut_init_pred - ut_init))
      elif order[0] == 3:
        IVloss = IVloss + tf.reduce_mean(tf.square(u_init_pred - u_init) + tf.square(ut_init_pred - ut_init) +
                               tf.square(utt_init_pred - utt_init))
      
      if order[1] == 1:
        IVloss = IVloss + tf.reduce_mean(tf.square(x_init_pred - x_init))
      elif order[1] == 2:
        IVloss = IVloss + tf.reduce_mean(tf.square(x_init_pred - x_init) + tf.square(xt_init_pred - xt_init))
      elif order[1] == 3:
        IVloss = IVloss + tf.reduce_mean(tf.square(x_init_pred - x_init) + tf.square(xt_init_pred - xt_init) +
                               tf.square(xtt_init_pred - xtt_init))
        
      if order[2] == 1:
        IVloss = IVloss + tf.reduce_mean(tf.square(y_init_pred - y_init))
      elif order[2] == 2:
        IVloss = IVloss + tf.reduce_mean(tf.square(y_init_pred - y_init) + tf.square(yt_init_pred - yt_init))
      elif order[2] == 3:
        IVloss = IVloss + tf.reduce_mean(tf.square(y_init_pred - y_init) + tf.square(yt_init_pred - yt_init) +
                               tf.square(ytt_init_pred - ytt_init))

      # Composite loss function
      loss = DEloss + gamma*IVloss

    grads = tape.gradient(loss, model.trainable_variables)
    return DEloss, IVloss, grads


def PINNtrain_3System_IVP(de_points, inits, order, t0, epochs, eqn, net_layers, net_units, model):
  """
  Main function called by PINNtrainSelect_Standard when solving system of 3 equations with soft constraint IVP.

  Args:
    de_points (list): Randomly sampled points for network to train with.
    inits (list): Inital values for network to learn
    order (list): Orders of equations to be solved
    t0 (float): First value in user input t_bdry
    epochs (int): Number of epochs for network to train
    eqn (list): Equations to solve. User input eqns
    net_layers (int): Number of internal layers of network
    net_units (int): Number of nodes for each internal layer
    model (PINN or None): User input model. Defaulted to None and model created in function
      with call to ode_ModelFuncs

  Returns:
    epoch_loss (list): Total loss over training of model
    ivp_loss (list): Inital Value loss over training of model
    de_loss (list): Differential equation loss over training of model
    model (PINN): Trained model to predict equation(s) over t

  Packages data correctly and calls train_network_3system_IVP in executing training routine, and handles
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
  if order[0] == 1:
    t_init, u_init = np.array([t0]).astype(np.float32), np.array([inits[0]]).astype(np.float32)
    if order[1] == 1:
      x_init = np.array([inits[1]]).astype(np.float32)
      if order[2] == 1:
        y_init = np.array([inits[2]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, x_init, y_init])
      elif order[2] == 2:
        y_init, yt_init = np.array([inits[2]]).astype(np.float32), np.array([inits[3]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, x_init, y_init, yt_init])
      elif order[2] == 3:
        y_init, yt_init = np.array([inits[2]]).astype(np.float32), np.array([inits[3]]).astype(np.float32)
        ytt_init = np.array([inits[4]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, x_init, y_init, yt_init, ytt_init])
    elif order[1] == 2:
      x_init, xt_init = np.array([inits[1]]).astype(np.float32), np.array([inits[2]]).astype(np.float32)
      if order[2] == 1:
        y_init = np.array([inits[3]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, x_init, xt_init, y_init])
      elif order[2] == 2:
        y_init, yt_init = np.array([inits[3]]).astype(np.float32), np.array([inits[4]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, x_init, xt_init, y_init, yt_init])
      elif order[2] == 3:
        y_init, yt_init = np.array([inits[3]]).astype(np.float32), np.array([inits[4]]).astype(np.float32)
        ytt_init = np.array([inits[5]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, x_init, xt_init, y_init, yt_init, ytt_init])
    elif order[1] == 3:
      x_init, xt_init = np.array([inits[1]]).astype(np.float32), np.array([inits[2]]).astype(np.float32)
      xtt_init = np.array([inits[3]]).astype(np.float32)
      if order[2] == 1:
        y_init = np.array([inits[4]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, x_init, xt_init, xtt_init, y_init])
      elif order[2] == 2:
        y_init, yt_init = np.array([inits[4]]).astype(np.float32), np.array([inits[5]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, x_init, xt_init, xtt_init, y_init, yt_init])
      elif order[2] == 3:
        y_init, yt_init = np.array([inits[4]]).astype(np.float32), np.array([inits[5]]).astype(np.float32)
        ytt_init = np.array([inits[6]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, x_init, xt_init, xtt_init, y_init, yt_init, ytt_init])
  elif order[0] == 2:
    t_init, u_init = np.array([t0]).astype(np.float32), np.array([inits[0]]).astype(np.float32)
    ut_init = np.array([inits[1]]).astype(np.float32)
    if order[1] == 1:
      x_init = np.array([inits[2]]).astype(np.float32)
      if order[2] == 1:
        y_init = np.array([inits[3]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, ut_init, x_init, y_init])
      elif order[2] == 2:
        y_init, yt_init = np.array([inits[3]]).astype(np.float32), np.array([inits[4]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, ut_init, x_init, y_init, yt_init])
      elif order[2] == 3:
        y_init, yt_init = np.array([inits[3]]).astype(np.float32), np.array([inits[4]]).astype(np.float32)
        ytt_init = np.array([inits[5]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, ut_init, x_init, y_init, yt_init, ytt_init])
    elif order[1] == 2:
      x_init, xt_init = np.array([inits[2]]).astype(np.float32), np.array([inits[3]]).astype(np.float32)
      if order[2] == 1:
        y_init = np.array([inits[4]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, ut_init, x_init, xt_init, y_init])
      elif order[2] == 2:
        y_init, yt_init = np.array([inits[4]]).astype(np.float32), np.array([inits[5]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, ut_init, x_init, xt_init, y_init, yt_init])
      elif order[2] == 3:
        y_init, yt_init = np.array([inits[4]]).astype(np.float32), np.array([inits[5]]).astype(np.float32)
        ytt_init = np.array([inits[6]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, ut_init, x_init, xt_init, y_init, yt_init, ytt_init])
    elif order[1] == 3:
      x_init, xt_init = np.array([inits[2]]).astype(np.float32), np.array([inits[3]]).astype(np.float32)
      xtt_init = np.array([inits[4]]).astype(np.float32)
      if order[2] == 1:
        y_init = np.array([inits[5]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, ut_init, x_init, xt_init, xtt_init, y_init])
      elif order[2] == 2:
        y_init, yt_init = np.array([inits[5]]).astype(np.float32), np.array([inits[6]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, ut_init, x_init, xt_init, xtt_init, y_init, yt_init])
      elif order[2] == 3:
        y_init, yt_init = np.array([inits[5]]).astype(np.float32), np.array([inits[6]]).astype(np.float32)
        ytt_init = np.array([inits[7]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, ut_init, x_init, xt_init, xtt_init, y_init, yt_init, ytt_init])
  elif order[0] == 3:
    t_init, u_init = np.array([t0]).astype(np.float32), np.array([inits[0]]).astype(np.float32)
    ut_init, utt_init = np.array([inits[1]]).astype(np.float32), np.array([inits[2]]).astype(np.float32)
    if order[1] == 1:
      x_init = np.array([inits[3]]).astype(np.float32)
      if order[2] == 1:
        y_init = np.array([inits[4]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, ut_init, utt_init, x_init, y_init])
      elif order[2] == 2:
        y_init, yt_init = np.array([inits[4]]).astype(np.float32), np.array([inits[5]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, ut_init, utt_init, x_init, y_init, yt_init])
      elif order[2] == 3:
        y_init, yt_init = np.array([inits[4]]).astype(np.float32), np.array([inits[5]]).astype(np.float32)
        ytt_init = np.array([inits[6]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, ut_init, utt_init, x_init, y_init, yt_init, ytt_init])
    elif order[1] == 2:
      x_init, xt_init = np.array([inits[3]]).astype(np.float32), np.array([inits[4]]).astype(np.float32)
      if order[2] == 1:
        y_init = np.array([inits[5]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, ut_init, utt_init, x_init, xt_init, y_init])
      elif order[2] == 2:
        y_init, yt_init = np.array([inits[5]]).astype(np.float32), np.array([inits[6]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, ut_init, utt_init, x_init, xt_init, y_init, yt_init])
      elif order[2] == 3:
        y_init, yt_init = np.array([inits[5]]).astype(np.float32), np.array([inits[6]]).astype(np.float32)
        ytt_init = np.array([inits[7]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, ut_init, utt_init, x_init, xt_init, y_init, yt_init, ytt_init])
    elif order[1] == 3:
      x_init, xt_init = np.array([inits[3]]).astype(np.float32), np.array([inits[4]]).astype(np.float32)
      xtt_init = np.array([inits[5]]).astype(np.float32)
      if order[2] == 1:
        y_init = np.array([inits[6]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, ut_init, utt_init, x_init, xt_init, xtt_init, y_init])
      elif order[2] == 2:
        y_init, yt_init = np.array([inits[6]]).astype(np.float32), np.array([inits[7]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, ut_init, utt_init, x_init, xt_init, xtt_init, y_init, yt_init])
      elif order[2] == 3:
        y_init, yt_init = np.array([inits[6]]).astype(np.float32), np.array([inits[7]]).astype(np.float32)
        ytt_init = np.array([inits[8]]).astype(np.float32)
        inits = np.column_stack([t_init, u_init, ut_init, utt_init, x_init, xt_init, xtt_init, y_init, yt_init, ytt_init])
  

  epoch_loss = np.zeros(epochs)
  ivp_loss = np.zeros(epochs)
  de_loss = np.zeros(epochs)
  nr_batches = 0

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
    model = build_model_system3_IVP(net_layers, net_units)

  # Main training loop
  for i in range(epochs):

    # Training for that epoch
    for (des, inits) in ds:

      nr_batches = 0

      # Train the network
      DEloss, IVloss, grads = train_network_3system_IVP(des, inits, order, model, gamma, eqn)

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