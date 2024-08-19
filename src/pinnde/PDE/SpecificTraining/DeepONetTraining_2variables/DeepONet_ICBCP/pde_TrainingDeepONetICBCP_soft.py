import tensorflow as tf
import numpy as np
import ast

@tf.function(reduce_retracing=True)
def train_network(pdes, inits, uIn, bcs, t_order, opt, model, equation, N_iv, extra_ders, boundary_type):


  # PDE points
  t_pde, x_pde = pdes[:,:1], pdes[:,1:2]
  t_init, x_init, u_init = inits[:,:1], inits[:,1:2], inits[:,2:3]
  if t_order == 2:
    ut_init = inits[:,3:4]
  elif t_order == 3:
    ut_init, utt_init = inits[:,3:4], inits[:,4:5]
  
  if boundary_type != "periodic_timeDependent":
    t_bc, x_bc, u_bc = bcs[:,:1], bcs[:,1:2], bcs[:,2:3]

  # Outer gradient for tuning network parameters
  with tf.GradientTape() as tape:

    with tf.GradientTape(persistent=True) as tape1:
      tape1.watch(t_pde), tape1.watch(x_pde)
      
      u = model([t_pde, x_pde, uIn])
      [ut, ux] = tape1.gradient(u, [t_pde, x_pde])
      utt = tape1.gradient(ut, t_pde)
      uttt = tape1.gradient(utt, t_pde)
      uxx = tape1.gradient(ux, x_pde)
      uxxx = tape1.gradient(uxx, x_pde)
      if extra_ders != None:
        for i in extra_ders:
            global utx, uxt
            uxt = utx = tape1.gradient(ux, t_pde)
            if ((i == "uxxt") or (i == "uxtx") or (i == "utxx")):
                global utxx, uxtx, uxxt
                uxxt = uxtx = utxx = tape1.gradient(utx, x_pde)
            if ((i == "uttx") or (i == "utxt") or (i == "uxtt")):
                global uttx, utxt, uxtt
                uxtt = utxt = uttx = tape1.gradient(uxt, t_pde)
    x = x_pde
    t = t_pde

    # Advection equation
    parse_tree = ast.parse(equation, mode="eval")
    eqn = eval(compile(parse_tree, "<string>", "eval"))


    # Loss
    PDloss = tf.reduce_mean(tf.square(eqn))

    # Define the initial value loss
    with tf.GradientTape(persistent=True) as tape2:
      tape2.watch(t_init), tape2.watch(x_init)
      u_init_pred = model([t_init, x_init, uIn])
      if t_order > 1:
        ut_init_pred = tape2.gradient(u_init_pred, t_init)
        if t_order > 2:
          utt_init_pred = tape2.gradient(ut_init_pred, t_init)

    if t_order == 1:
      IVloss = tf.reduce_mean(tf.square(u_init_pred - u_init))
    elif t_order == 2:
      IVloss = tf.reduce_mean(tf.square(u_init_pred - u_init) + tf.square(ut_init_pred - ut_init))
    elif t_order == 3:
      IVloss = tf.reduce_mean(tf.square(u_init_pred - u_init) + tf.square(ut_init_pred - ut_init) +
                                tf.square(utt_init_pred - utt_init))
      
    #Define BC losss
    if boundary_type == "dirichlet_timeDependent":
      tbc_left = t_bc[:N_iv]
      tbc_right = t_bc[N_iv:]
      xbc_left = x_bc[:N_iv]
      xbc_right = x_bc[N_iv:]
      uleft = u_bc[:N_iv]
      uright = u_bc[N_iv:]
      uleft_bc_pred = model([tbc_left, xbc_left, uIn])
      uright_bc_pred = model([tbc_right, xbc_right, uIn])
      BCloss = tf.reduce_mean(tf.square(uleft-uleft_bc_pred) + tf.square(uright-uright_bc_pred))
      
      loss = PDloss + IVloss + BCloss

      # Model gradients and update
      grads = tape.gradient(loss, model.trainable_variables)
      opt.apply_gradients(zip(grads, model.trainable_variables))

      return loss, PDloss, IVloss, BCloss
    
    elif boundary_type == "neumann_timeDependent":
      tbc_left = t_bc[:N_iv]
      tbc_right = t_bc[N_iv:]
      xbc_left = x_bc[:N_iv]
      xbc_right = x_bc[N_iv:]
      uleft = u_bc[:N_iv]
      uright = u_bc[N_iv:]
      with tf.GradientTape(persistent=True) as tape3:
        tape3.watch(tbc_left), tape3.watch(tbc_right), tape3.watch(xbc_left), tape3.watch(xbc_right)
        uleft_bc_pred = model([tbc_left, xbc_left, uIn])
        uright_bc_pred = model([tbc_right, xbc_right, uIn])
        duleft_bc_pred = tape3.gradient(uleft_bc_pred, xbc_left)
        duright_bc_pred = tape3.gradient(uright_bc_pred, xbc_right)
      BCloss = tf.reduce_mean(tf.square(uleft-duleft_bc_pred) + tf.square(uright-duright_bc_pred))

      loss = PDloss + IVloss + BCloss

      # Model gradients and update
      grads = tape.gradient(loss, model.trainable_variables)
      opt.apply_gradients(zip(grads, model.trainable_variables))

      return loss, PDloss, IVloss, BCloss
      
    else:
      loss = PDloss + IVloss

      # Model gradients and update
      grads = tape.gradient(loss, model.trainable_variables)
      opt.apply_gradients(zip(grads, model.trainable_variables))

      return loss, PDloss, IVloss, IVloss

def train(pde_points, inits, u_sensors, setup_boundaries, t_order, epochs, model, eqn, N_iv, extra_ders):
  
  boundary_type = setup_boundaries[0]

  # Number of ICs and total number of PDE collocation points
  N = pde_points.shape[0]
  if boundary_type != "periodic_timeDependent":
    bcs = setup_boundaries[1]
  else:
    bcs = 0

  # Batch size for PDE points
  batch_size = N_iv
  
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
      loss, PDloss, IVloss, BCloss = train_network(pdes, inits, usensor, bcs, t_order, opt, model, eqn, N_iv, extra_ders, boundary_type)

      epoch_loss[i] += loss
      pde_loss[i] += PDloss
      iv_loss[i] += IVloss
      if boundary_type != "periodic_timeDependent":
        bc_loss[i] += BCloss
      nr_batches += 1

    # Get total epoch loss
    epoch_loss[i] /= nr_batches

    if (np.mod(i, 100)==0):
      if boundary_type == "periodic_timeDependent":
        BCloss = 0+0*IVloss
      print("PDE loss, IV loss, BC loss in {}th epoch: {: 6.4f}, {: 6.4f}, {: 6.4f}.".format(i, PDloss.numpy(), IVloss.numpy(), BCloss.numpy()))


  return epoch_loss, iv_loss, bc_loss, pde_loss, model