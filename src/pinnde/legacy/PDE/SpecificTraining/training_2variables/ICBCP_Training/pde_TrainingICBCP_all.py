import numpy as np
import tensorflow as tf
import ast
tf.keras.backend.set_floatx('float64')

@tf.function
def trainStep(pdes, inits, bcs, model, t_order, eqnparam, extra_ders, boundary_type):

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
        # # Inner gradient for derivatives of u wrt x and t
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(t_pde), tape1.watch(x_pde)
            u = model([t_pde, x_pde])
            [ut, ux] = tape1.gradient(u, [t_pde, x_pde])
            uxx = tape1.gradient(ux, x_pde)
            uxxx = tape1.gradient(uxx, x_pde)
            utt = tape1.gradient(ut, t_pde)
            uttt = tape1.gradient(utt, t_pde)
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
        t = t_pde
        x = x_pde

        parse_tree = ast.parse(eqnparam, mode="eval")
        eqn = eval(compile(parse_tree, "<string>", "eval"))
        
        # Define the PDE loss
        PDEloss = tf.reduce_mean(tf.square(eqn))

        #Define BC losss
        if boundary_type == "dirichlet_timeDependent":
            u_bc_pred = model([t_bc, x_bc])
            BCloss = tf.reduce_mean(tf.square(u_bc-u_bc_pred))

        elif boundary_type == "neumann_timeDependent":
            with tf.GradientTape() as tape3:
                tape3.watch(t_bc), tape3.watch(x_bc)
                ux_bc_pred = model([t_bc, x_bc])
                dux_bc_pred = tape3.gradient(ux_bc_pred, x_bc)
            BCloss = tf.reduce_mean(tf.square(u_bc-dux_bc_pred))
        

        # Define the initial value loss
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(t_init), tape2.watch(x_init)
            u_init_pred = model([t_init, x_init])
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

        # Global loss
        if boundary_type != "periodic_timeDependent":
            loss = PDEloss + IVloss + BCloss
            grads = tape.gradient(loss, model.trainable_variables)
            return PDEloss, IVloss, BCloss, grads
        else:
            loss = PDEloss + IVloss
            grads = tape.gradient(loss, model.trainable_variables)
            return PDEloss, IVloss, IVloss, grads


def PINNtrain(pde_points, init_points, t_order, setup_boundaries, epochs, eqn, N_pde, N_iv, model, extra_ders, adapt):
    boundary_type = setup_boundaries[0]

    # Optimizer to be used
    lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, epochs, 1e-4)
    opt = tf.keras.optimizers.Adam(lr)


    bs_pdes, bs_inits = N_pde//10, N_iv//10

    ds_pde = tf.data.Dataset.from_tensor_slices(pde_points)
    ds_pde = ds_pde.cache().shuffle(N_pde).batch(bs_pdes)

    ds_init = tf.data.Dataset.from_tensor_slices(init_points)
    ds_init = ds_init.cache().shuffle(N_iv).batch(bs_inits)

    if boundary_type != "periodic_timeDependent":
        N_bc = setup_boundaries[2]
        bcs_points = setup_boundaries[1]
        bs_bc = 2*N_bc//10
        ds_bc = tf.data.Dataset.from_tensor_slices(bcs_points)
        ds_bc = ds_bc.cache().shuffle(N_bc).batch(bs_bc)
        ds = tf.data.Dataset.zip((ds_pde, ds_init, ds_bc))
    
    else:  
        ds = tf.data.Dataset.zip((ds_pde, ds_init, ds_init)) #second init just to match the bc needed in loop
        
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    epoch_loss = np.zeros(epochs)
    iv_loss = np.zeros(epochs)
    pde_loss = np.zeros(epochs)
    bc_loss = np.zeros(epochs)

    # Main training loop
    for i in range(epochs):

        n_batches = 0
        for (pdes, inits, bcs) in ds:

            PDEloss, IVloss, BCloss, grads = trainStep(pdes, inits, bcs, model, t_order, eqn, extra_ders, boundary_type)

            # Gradient step
            opt.apply_gradients(zip(grads, model.trainable_variables))
            # One more batch done
            n_batches += 1
            if boundary_type != "periodic_timeDependent":
                epoch_loss[i] += PDEloss + BCloss + IVloss
                bc_loss[i] += BCloss
            else:
                epoch_loss[i] += PDEloss + IVloss

            pde_loss[i] += PDEloss
            iv_loss[i] += IVloss

        epoch_loss[i] /= n_batches

    
        if (np.mod(i, 100)==0):
            if boundary_type == "periodic_timeDependent":
                BCloss = 0+0*IVloss
            print("PDE loss, IV loss, BC loss in {}th epoch: {: 6.4f}, {: 6.4f}, {: 6.4f}.".format(i, PDEloss.numpy(), IVloss.numpy(), BCloss.numpy()))

    return epoch_loss, iv_loss, bc_loss, pde_loss, model