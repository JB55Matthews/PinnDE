import numpy as np
import tensorflow as tf
import ast
tf.keras.backend.set_floatx('float64')

@tf.function
def trainStep(pdes, model, eqnparam):

    t_pde, x_pde, y_pde = pdes[:,:1], pdes[:,1:2], pdes[:,2:3]

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
        

        # Global loss
        loss = PDEloss + 0

    # Compute the gradient of the global loss wrt the model parameters
    grads = tape.gradient(loss, model.trainable_variables)

    return PDEloss, grads

def PINNtrain(pde_points, setup_boundries, epochs, eqn, N_pde, N_iv, model):

    # Optimizer to be used
    lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, epochs, 1e-4)
    opt = tf.keras.optimizers.Adam(lr)

    bs_pdes, bs_inits = N_pde//10, N_iv//10

    ds_pde = tf.data.Dataset.from_tensor_slices(pde_points)
    ds_pde = ds_pde.cache().shuffle(N_pde).batch(bs_pdes)

    ds = tf.data.Dataset.zip((ds_pde))
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    epoch_loss = np.zeros(epochs)
    iv_loss = np.zeros(epochs)
    pde_loss = np.zeros(epochs)
    bc_loss = np.zeros(epochs)

    # Main training loop
    for i in range(epochs):

        n_batches = 0
        for (pdes) in ds:

            PDEloss, grads = trainStep(pdes, model, eqn)

            # Gradient step
            opt.apply_gradients(zip(grads, model.trainable_variables))
            # One more batch done
            n_batches += 1
            epoch_loss[i] += PDEloss
            pde_loss[i] += PDEloss
    

        epoch_loss[i] /= n_batches

    
        if (np.mod(i, 100)==0):
            print("PDE loss, IV loss in {}th epoch: {: 6.4f}.".format(i, PDEloss.numpy()))

    return epoch_loss, iv_loss, bc_loss, pde_loss, model