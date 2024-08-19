import numpy as np
import tensorflow as tf
import ast
tf.keras.backend.set_floatx('float64')

@tf.function
def trainStep(pdes, bcs, model, eqnparam, extra_ders, boundary_type):

    x_pde, y_pde = pdes[:,:1], pdes[:,1:2]
    if boundary_type == "dirichlet_timeIndependent":
        x_bc, y_bc, u_bc = bcs[:,:1], bcs[:,1:2], bcs[:,2:3]
    elif boundary_type == "neumann_timeIndependent":
        x_bound, y_bound, x_bc, y_bc, ux_bc, uy_bc = bcs[:,:1], bcs[:,1:2], bcs[:,2:3], bcs[:,3:4], bcs[:,4:5], bcs[:,5:6]
    
  # Outer gradient for tuning network parameters
    with tf.GradientTape() as tape:
        # # Inner gradient for derivatives of u wrt x and t
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x_pde), tape1.watch(y_pde)
            u = model([x_pde, y_pde])
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

        y = y_pde
        x = x_pde

        parse_tree = ast.parse(eqnparam, mode="eval")
        eqn = eval(compile(parse_tree, "<string>", "eval"))
        
        # Define the PDE loss
        PDEloss = tf.reduce_mean(tf.square(eqn))

        #Define BC losss
        if boundary_type == "dirichlet_timeIndependent":
            u_bc_pred = model([x_bc, y_bc])
            BCloss = tf.reduce_mean(tf.square(u_bc-u_bc_pred))
            
        elif boundary_type == "neumann_timeIndependent":
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x_bc), tape2.watch(y_bc), tape2.watch(x_bound), tape2.watch(y_bound)
                ux_bc_pred = model([x_bound, y_bc])
                uy_bc_pred = model([x_bc, y_bound])
                dux_bc_pred = tape2.gradient(ux_bc_pred, x_bound)
                duy_bc_pred = tape2.gradient(uy_bc_pred, y_bound)
            BCloss = tf.reduce_mean(tf.square(ux_bc-dux_bc_pred)+tf.square(uy_bc-duy_bc_pred))

        # Global loss
        loss = PDEloss + BCloss

    # Compute the gradient of the global loss wrt the model parameters
    grads = tape.gradient(loss, model.trainable_variables)

    return PDEloss, BCloss, grads

def PINNtrain(pde_points, setup_boundaries, epochs, eqn, N_pde, model, extra_ders):
    

    # Optimizer to be used
    lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, epochs, 1e-4)
    opt = tf.keras.optimizers.Adam(lr)
    N_bc = setup_boundaries[2]
    bcs_points = setup_boundaries[1]
    boundary_type = setup_boundaries[0]

    bs_pdes, bs_bc = N_pde//10, 4*N_bc//10

    ds_pde = tf.data.Dataset.from_tensor_slices(pde_points)
    ds_pde = ds_pde.cache().shuffle(N_pde).batch(bs_pdes)

    ds_bc = tf.data.Dataset.from_tensor_slices(bcs_points)
    ds_bc = ds_bc.cache().shuffle(N_bc).batch(bs_bc)

    ds = tf.data.Dataset.zip((ds_pde, ds_bc))
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    epoch_loss = np.zeros(epochs)
    iv_loss = np.zeros(epochs)
    pde_loss = np.zeros(epochs)
    bc_loss = np.zeros(epochs)

    # Main training loop
    for i in range(epochs):

        n_batches = 0
        for (pdes, bcs) in ds:

            PDEloss, BCloss, grads = trainStep(pdes, bcs, model, eqn, extra_ders, boundary_type)

            # Gradient step
            opt.apply_gradients(zip(grads, model.trainable_variables))
            ## One more batch done
            n_batches += 1
            epoch_loss[i] += PDEloss + BCloss
            pde_loss[i] += PDEloss
            bc_loss[i] += BCloss

        epoch_loss[i] /= n_batches

    
        if (np.mod(i, 100)==0):
            print("PDE loss, BC loss in {}th epoch: {: 6.4f}, {: 6.4f}.".format(i, PDEloss.numpy(), BCloss.numpy()))

    return epoch_loss, iv_loss, bc_loss, pde_loss, model