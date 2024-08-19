import numpy as np
import tensorflow as tf
import ast
tf.keras.backend.set_floatx('float64')

@tf.function
def trainStep(pdes, model, eqnparam, extra_ders):
    """
    Function which does the training for a single epoch

    Args:
        pdes (list): Sampled pde points network uses to train
        model (PINN): Model to train
        eqnparam (string): Equation to solve.
        extra_ders (list): Extra derivatives needed to be computed for user equation


    Generates derivatives of model using automatic differentiation. Computes
    mean squared error of loss along pdes points.

    Returns:
        PDEloss (list): Loss of training network to match function along pdes points
        grads (array): Gradients of network for optimization
    """

    x_pde, y_pde = pdes[:,:1], pdes[:,1:2]
    
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

        # Global loss
        loss = PDEloss + 0

    # Compute the gradient of the global loss wrt the model parameters
    grads = tape.gradient(loss, model.trainable_variables)

    return PDEloss, grads

def PINNtrain(pde_points, setup_boundaries, epochs, eqn, N_pde, model, extra_ders):
    """
    Main function called by PINNtrainSelect_xy when solving equation in xy with periodic boundaries.

    Args:
        pde_points (list): pde_points returned from defineCollocationPoints_xy()
        setup_boundaries (boundary): boundary conditions set up from return of pde_Boundaries_2var call.
        epochs (int): Number of epochs model gets trained for
        eqn (string): Equation to be solved 
        N_pde (int): Number of randomly sampled collocation points along t and x which PINN uses in training.
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

            PDEloss, grads = trainStep(pdes, model, eqn, extra_ders)

            # Gradient step
            opt.apply_gradients(zip(grads, model.trainable_variables))
            ## One more batch done
            n_batches += 1
            epoch_loss[i] += PDEloss
            pde_loss[i] += PDEloss

        epoch_loss[i] /= n_batches

    
        if (np.mod(i, 100)==0):
            print("PDE loss in {}th epoch: {: 6.4f}.".format(i, PDEloss.numpy()))

    return epoch_loss, iv_loss, bc_loss, pde_loss, model