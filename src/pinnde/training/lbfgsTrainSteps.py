import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def value_and_gradients_generator(model, lossfn, lossfn_data, eqns, extras):
    """
    Generator which returns a function to compute the value_and_graidents function in tfp.optimizer.lbfgs_minimize

    Args:
        model (model): TensorFlow model to train.
        lossfn (function): Loss function taken from other train steps to minimize
        lossfn_data (list): List of data required for loss function.
        eqns (list): List of equations to learn.
        extras (list): List of extra data required for loss function depending on trainign routine.

    Returns:
        (function): Value and gradients function.
    """
  # (eqns, clps, bcs, ics, network, boundary, t_orders, constraint)

    shapes = []
    for i in range(len(model.trainable_variables)):
        shapes.append(model.trainable_variables[i].shape)
    n_tensors = len(shapes)
    count = 0
    idx = []
    partition = []
    for i, shape in enumerate(shapes):
        n = np.prod(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        partition.extend([i]*n)
        count += n
    partition = tf.constant(partition)

    @tf.function
    def assign_new_model_parameters(flat_params):

        params = tf.dynamic_partition(flat_params, partition, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # @tf.function
    def value_and_gradients_func(flat_params):

        with tf.GradientTape() as tape:
            assign_new_model_parameters(flat_params)
            # CLPloss, BCloss, ICloss, _ = loss(eqns, *lossfn_data, model, *extras)
            lossesgrads = lossfn(eqns, *lossfn_data, model, *extras)
            # loss_value = CLPloss + BCloss + ICloss
            # print(lossesgrads)
            losses = lossesgrads[:-1]
            loss_value = tf.add_n(losses)

        grads = tape.gradient(loss_value, model.trainable_variables)
        # ensure all are non-None
        # for i, g in enumerate(grads):
        #     if g is None:
        #         print(f"[WARNING] Gradient is None at variable {model.trainable_variables[i].name}")
        #         grads[i] = tf.zeros_like(model.trainable_variables[i])

        grads = tf.dynamic_stitch(idx, grads)

        if np.mod(value_and_gradients_func.iter, 50) == 0:
          tf.print("Inner iteration:", value_and_gradients_func.iter, "loss:", loss_value)
        value_and_gradients_func.iter.assign_add(1)
        
        tf.py_function(value_and_gradients_func.history.append, inp=[loss_value], Tout=[])

        return loss_value, grads

    value_and_gradients_func.iter = tf.Variable(0)
    value_and_gradients_func.idx = idx
    value_and_gradients_func.partition = partition
    value_and_gradients_func.shapes = shapes
    value_and_gradients_func.assign_new_model_parameters = assign_new_model_parameters
    value_and_gradients_func.history = []

    return value_and_gradients_func


def lbfgsTrain(model, eqns, epochs, lossfn, lossfn_data, extras):
  """
    Initiaalizes training for tfp.optimizer.lbfgs_minimize training

    Args:
        model (model): TensorFlow model to train.
        eqns (list): List of equations to learn.
        epochs (int): Outer iterations for L-BFGS to take.
        lossfn (function): Loss function taken from other train steps to minimize
        lossfn_data (list): List of data required for loss function.
        extras (list): List of extra data required for loss function depending on trainign routine.

    Returns:
        (list): Trainign loss of inner iterations.
        (int): Number of inner training iterations.
    """
  
  func = value_and_gradients_generator(model, lossfn, lossfn_data, eqns, extras)
  init_params = tf.dynamic_stitch(func.idx, model.trainable_variables)
  # train the model with L-BFGS solver
  results = tfp.optimizer.lbfgs_minimize(
      value_and_gradients_function=func, initial_position=init_params, max_iterations=epochs)

  func.assign_new_model_parameters(results.position)

  return func.history, len(func.history)