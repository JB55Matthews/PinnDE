import numpy as np
import tensorflow as tf
import ast
tf.keras.backend.set_floatx('float64')

@tf.function
def trainStep(pdes, model, t_order, eqnparam, extra_ders):
    """
    Function which does the training for a single epoch

    Args:
        pdes (list): Sampled pde points network uses to train
        model (PINN): Model to train
        t_order (int): Order of equation to solve
        eqnparam (string): Equation to solve.
        extra_ders (list): Extra derivatives needed to be computed for user equation


    Generates derivatives of model using automatic differentiation. Computes
    mean squared error of loss along pdes points.

    Returns:
        PDEloss (list): Loss of training network to match function along pdes points
        IVloss (list): Loss of training network to match initial values
        BCloss (list): Loss of training network to match boundary values
        grads (array): Gradients of network for optimization
    """

    t_pde, x_pde = pdes[:,:1], pdes[:,1:2]

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


        # Global loss
        loss = PDEloss

    # Compute the gradient of the global loss wrt the model parameters
    grads = tape.gradient(loss, model.trainable_variables)

    return PDEloss, grads

def PINNtrain(pde_points, init_points, t_order, setup_boundries, epochs, eqn, N_pde, N_iv, model, extra_ders):
    """
    Main function called by PINNtrainSelect_tx when solving equation in tx with hard constraint.

    Args:
        pde_points (list): pde_points returned from defineCollocationPoints_tx()
        init_points (list): inits returned from defineCollocationPoints_tx()
        t_order (int): Order of t in equation
        setup_boundries (boundary): boundary conditions set up from return of pde_Boundries_2var call.
        epochs (int): Number of epochs model gets trained for
        eqn (string): Equation to be solved 
        N_pde (int): Number of randomly sampled collocation points along t and x which PINN uses in training.
        N_iv (int): Number of randomly sampled collocation points along inital t which PINN uses in training.
        model (PINN): Model created from pde_ModelFuncs_2var or input model
        extra_ders (list): Extra derivatives needed to be computed for user equation

    Returns:
        epoch_loss (list): Total loss over training of model
        iv_loss (list): Inital value loss over training of model
        bc_loss (list): Boundary condition loss over training of model
        pde_loss (list): Differential equation loss over training of model
        model (PINN): Trained model to predict equation solution

    Packages data correctly and calls trainStep in executing training routine, and handles
    optimization of the network.
    """

    # Optimizer to be used
    lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, epochs, 1e-4)
    opt = tf.keras.optimizers.Adam(lr)

    bs_pdes = N_pde//10

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

            PDEloss, grads = trainStep(pdes, model, t_order, eqn, extra_ders)

            # Gradient step
            opt.apply_gradients(zip(grads, model.trainable_variables))
            # One more batch done
            n_batches += 1
            epoch_loss[i] += PDEloss
            pde_loss[i] += PDEloss

        epoch_loss[i] /= n_batches

    
        if (np.mod(i, 100)==0):
            print("PDE loss in {}th epoch: {: 6.4f}.".format(i, PDEloss.numpy()))

    return epoch_loss, iv_loss, bc_loss, pde_loss, model