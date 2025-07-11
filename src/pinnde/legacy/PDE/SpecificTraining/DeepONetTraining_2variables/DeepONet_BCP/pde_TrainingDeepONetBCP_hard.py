import tensorflow as tf
import numpy as np
import ast

@tf.function(reduce_retracing=True)
def train_network(pdes, uIn, opt, model, equation, extra_ders):
  """
  Function which does the training for a single epoch

  Args:
    pdes (list): Sampled pde points network uses to train
    uIn (list): Sensor points network uses to train
    opt (Optimizer): Keras.Optimizer.Adam optimizer
    model (DeepONet): Model to train
    equation (string): Equation to solve
    extra_ders (list): Extra derivatives needed to be computed for user equation


  Generates derivatives of model using automatic differentiation. Computes
  mean squared error of loss along pdes points. Also handles
  optimization of the network.

  Returns:
    loss (list): Total loss of training network during epoch
    PDloss (list): Loss of training network to match function along pdes points
    """

  # PDE points
  x_pde, y_pde = pdes[:,:1], pdes[:,1:2]

  # Outer gradient for tuning network parameters
  with tf.GradientTape() as tape:

    with tf.GradientTape(persistent=True) as tape1:
      tape1.watch(x_pde), tape1.watch(y_pde)
    
      u = model([x_pde, y_pde, uIn])
      [ux, uy] = tape1.gradient(u, [x_pde, y_pde])
      uxx = tape1.gradient(ux, x_pde)
      uxxx = tape1.gradient(uxx, x_pde)
      uyy = tape1.gradient(uy, y_pde)
      uyyy = tape1.gradient(uyy, y_pde)
      if extra_ders != None:
        for i in extra_ders:
            global uxy, uyx
            uxy = uyx = tape1.gradient(ux, y_pde)
            if ((i == "uxxy") or (i == "uxyx") or (i == "uyxx")):
                global uyxx, uxyx, uxxy
                uxxy = uxyx = uyxx = tape1.gradient(uyx, x_pde)
            if ((i == "uyyx") or (i == "uyxy") or (i == "uxyy")):
                global uyyx, uyxy, uxyy
                uxyy = uyxy = uyyx = tape1.gradient(uxy, y_pde)
    x = x_pde
    y = y_pde

    # Advection equation
    parse_tree = ast.parse(equation, mode="eval")
    eqn = eval(compile(parse_tree, "<string>", "eval"))


    # Loss
    PDloss = tf.reduce_mean(tf.square(eqn))
      
    loss = PDloss + 0

  # Model gradients and update
  grads = tape.gradient(loss, model.trainable_variables)
  opt.apply_gradients(zip(grads, model.trainable_variables))

  return loss, PDloss

def train(pde_points, u_sensors, epochs, model, eqn, N_bc, extra_ders):
  """
  Main function called by PINNtrainSelect_DeepONet_xy when solving equation in xy with hard constraints or periodic boundaries
  (as those equations function the same, no learning of any boundary conditions).

  Args:
    pde_points (list): pde_points returned from defineCollocationPoints_DON_xy()
    u_sensors (list): usensors returned from defineCollocationPoints_DON_xy()
    epochs (int): Number of epochs model gets trained for
    model (DeepONet): Model created from pde_DeepONetModelFuncs_2var or input model
    eqn (string): Equation to be solved 
    N_bc (int): Number of randomly sampled collocation points along boundaries which DeepONet uses in training.
    extra_ders (list): Extra derivatives needed to be computed for user equation

  Returns:
    epoch_loss (list): Total loss over training of model
    iv_loss (list): Inital value loss over training of model
    bc_loss (list): Boundary condition loss over training of model
    pde_loss (list): Differential equation loss over training of model
    model (DeepONet): Trained model to predict equation solution

  Packages data correctly and calls train_network in executing training routine.
    """

  # Number of ICs and total number of PDE collocation points
  N = pde_points.shape[0]

  # Batch size for PDE points
  batch_size = N_bc
  
  ds_u = tf.data.Dataset.from_tensor_slices(u_sensors)
  ds_pde = tf.data.Dataset.from_tensor_slices(pde_points)

  ds = tf.data.Dataset.zip((ds_pde, ds_u))
  ds = ds.cache().shuffle(N).batch(batch_size)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  # Optimizer and learning rate to be used
  lr = 1e-3
  opt = tf.keras.optimizers.Adam(lr)

#   lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#         1.2e-2, decay_steps = 200, decay_rate = 0.95, staircase=False)
#   opt = tf.keras.optimizers.Adam(lr_schedule, beta_1 = 0.95, beta_2 = 0.99)


  epoch_loss = np.zeros(epochs)
  iv_loss = np.zeros(epochs)
  pde_loss = np.zeros(epochs)
  bc_loss = np.zeros(epochs)

  # Main training loop
  for i in range(epochs):

    nr_batches = 0

    # Training for that epoch
    for (pdes, usensor) in ds:

      # Train the network
      loss, PDloss = train_network(pdes, usensor, opt, model, eqn, extra_ders)

      epoch_loss[i] += loss
      pde_loss[i] += PDloss
      nr_batches += 1

    # Get total epoch loss
    epoch_loss[i] /= nr_batches

    if (np.mod(i, 100)==0):
      print(f"PDE loss {i}th epoch: {PDloss: 6.4f}.")

  return epoch_loss, iv_loss, bc_loss, pde_loss, model