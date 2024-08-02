import numpy as np
import tensorflow as tf
import ast
tf.keras.backend.set_floatx('float64')

@tf.function
def trainStep(pdes, inits, model, t_order, eqnparam):

    t_pde, x_pde, y_pde = pdes[:,:1], pdes[:,1:2], pdes[:,2:3]
    t_init, x_init, y_init, u_init = inits[:,:1], inits[:,1:2], inits[:,2:3], inits[:,3:4]
    if t_order == 2:
        ut_init = inits[:,3:4]
    elif t_order == 3:
        ut_init, utt_init = inits[:,3:4], inits[:,4:5]

  # Outer gradient for tuning network parameters
    with tf.GradientTape() as tape:
        # # Inner gradient for derivatives of u wrt x and t
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(t_pde), tape1.watch(x_pde), tape1.watch(y_pde)
            u = model([t_pde, x_pde, y_pde])
            [ut, ux, uy] = tape1.gradient(u, [t_pde, x_pde, y_pde])
            uxx = tape1.gradient(ux, x_pde)
            uxxx = tape1.gradient(uxx, x_pde)
            utt = tape1.gradient(ut, t_pde)
            uttt = tape1.gradient(utt, t_pde)
            uyy = tape1.gradient(uy, y_pde)
            uyyy = tape1.gradient(uyy, y_pde)



        t = t_pde
        x = x_pde
        y = y_pde

        parse_tree = ast.parse(eqnparam, mode="eval")
        eqn = eval(compile(parse_tree, "<string>", "eval"))
        
        # Define the PDE loss
        PDEloss = tf.reduce_mean(tf.square(eqn))

        # Define the initial value loss
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(t_init), tape2.watch(x_init), tape2.watch(y_init)
            u_init_pred = model([t_init, x_init, y_init])
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
        loss = PDEloss + IVloss

    # Compute the gradient of the global loss wrt the model parameters
    grads = tape.gradient(loss, model.trainable_variables)

    return PDEloss, IVloss, grads

def PINNtrain(pde_points, init_points, t_order, setup_boundries, epochs, eqn, N_pde, N_iv, model):

    # Optimizer to be used
    lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, epochs, 1e-4)
    opt = tf.keras.optimizers.Adam(lr)

    bs_pdes, bs_inits = N_pde//10, N_iv//10

    ds_pde = tf.data.Dataset.from_tensor_slices(pde_points)
    ds_pde = ds_pde.cache().shuffle(N_pde).batch(bs_pdes)

    ds_init = tf.data.Dataset.from_tensor_slices(init_points)
    ds_init = ds_init.cache().shuffle(N_iv).batch(bs_inits)

    ds = tf.data.Dataset.zip((ds_pde, ds_init))
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    epoch_loss = np.zeros(epochs)
    iv_loss = np.zeros(epochs)
    pde_loss = np.zeros(epochs)
    bc_loss = np.zeros(epochs)

    # Main training loop
    for i in range(epochs):

        n_batches = 0
        for (pdes, inits) in ds:

            PDEloss, IVloss, grads = trainStep(pdes, inits, model, t_order, eqn)

            # Gradient step
            opt.apply_gradients(zip(grads, model.trainable_variables))
            # One more batch done
            n_batches += 1
            epoch_loss[i] += PDEloss + IVloss
            pde_loss[i] += PDEloss
            iv_loss[i] += IVloss

        epoch_loss[i] /= n_batches

    
        if (np.mod(i, 100)==0):
            print("PDE loss, IV loss in {}th epoch: {: 6.4f}, {: 6.4f}.".format(i, PDEloss.numpy(), IVloss.numpy()))

    return epoch_loss, iv_loss, bc_loss, pde_loss, model