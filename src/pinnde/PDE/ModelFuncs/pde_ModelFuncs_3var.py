import tensorflow as tf
import numpy as np

def select_model_txy(t_bdry, x_bdry, y_bdry, t_order, inital_t, net_layers, net_units, constraint, setup_boundaries):
  boundary_type = setup_boundaries[0]

  if boundary_type == "periodic_timeDependent":
    if constraint == "soft":
      model = build_model_periodic_txy(t_bdry, net_layers, net_units)
    elif constraint == "hard":
      if t_order == 1:
        model = build_model_periodic_hardconstraint1_txy(t_bdry, inital_t[0], net_layers, net_units)
  #     elif t_order == 2:
  #       model = build_model_periodic_hardconstraint2_tx(t_bdry, inital_t[0], inital_t[1], net_layers, net_units)
  #     elif t_order == 3:
  #       model = build_model_periodic_hardconstraint3_tx(t_bdry, inital_t[0], inital_t[1], inital_t[2], net_layers, net_units)
                      
  elif boundary_type == "dirichlet_timeDependent":
    if constraint == "soft":
      model = build_model_soft_txy(t_bdry, x_bdry, y_bdry, net_layers, net_units)
  #   elif constraint=="hard":
  #     if t_order == 1:
  #       model = build_model_dirichlet_hardconstraint1_tx(t_bdry, x_bdry, inital_t[0], net_layers, net_units, setup_boundaries[3], setup_boundaries[4])

  # elif boundary_type == "neumann_timeDependent":
  #   if constraint == "soft":
  #     model = build_model_xy(t_bdry, x_bdry, net_layers, net_units)
      
  return model

class Periodic(tf.keras.layers.Layer):
  def __init__(self):
    super(Periodic, self).__init__()

  def call(self, inputs):
    return tf.concat((tf.cos(np.pi*inputs), tf.sin(np.pi*inputs)), axis=1)
  
# Define the normalization layer
class Normalize(tf.keras.layers.Layer):
  def __init__(self, xmin, xmax, name=None, **kwargs):
    super(Normalize, self).__init__(name=name)
    self.xmin = xmin
    self.xmax = xmax
    super(Normalize, self).__init__(**kwargs)

  def call(self, inputs):
    return 2.0*(inputs-self.xmin)/(self.xmax-self.xmin)-1.0

  def get_config(self):
    config = super(Normalize, self).get_config()
    config.update({'xmin': self.xmin, 'xmax': self.xmax})
    return config

# Define the network
def build_model_periodic_txy(t, n_layers, n_units):

  # Define the network
  inp1 = tf.keras.layers.Input(shape=(1,))
  b1 = Normalize(t[0], t[1])(inp1)

  inp2 = tf.keras.layers.Input(shape=(1,))
  b2 = Periodic()(inp2)

  inp3 = tf.keras.layers.Input(shape=(1,))
  b3 = Periodic()(inp3)

  b = tf.keras.layers.Concatenate()([b1, b2, b3])

  for i in range(n_layers):
    b = tf.keras.layers.Dense(n_units, activation='tanh')(b)
  out = tf.keras.layers.Dense(1, activation='linear')(b)

  model = tf.keras.models.Model([inp1, inp2, inp3], out)
  model.summary()

  return model

def build_model_test(t, x, y, n_layers, n_units):

  # Define the network
  inp1 = tf.keras.layers.Input(shape=(1,))
  b1 = Normalize(t[0], t[1])(inp1)

  inp2 = tf.keras.layers.Input(shape=(1,))
  b2 = Normalize(x[0], x[1])(inp2)

  inp3 = tf.keras.layers.Input(shape=(1,))
  b3 = Normalize(y[0], y[1])(inp3)

  b = tf.keras.layers.Concatenate()([b1, b2, b3])

  for i in range(n_layers):
    b = tf.keras.layers.Dense(n_units, activation='tanh')(b)
  out = tf.keras.layers.Dense(1, activation='linear')(b)

  model = tf.keras.models.Model([inp1, inp2, inp3], out)
  model.summary()

  return model

def build_model_periodic_hardconstraint1_txy(t, u0, n_layers, n_units):

  # Define the network
  inp1 = tf.keras.layers.Input(shape=(1,))
  b1 = Normalize(t[0], t[1])(inp1)

  inp2 = tf.keras.layers.Input(shape=(1,))
  b2 = Periodic()(inp2)

  inp3 = tf.keras.layers.Input(shape=(1,))
  b3 = Periodic()(inp3)

  b = tf.keras.layers.Concatenate()([b1, b2, b3])

  for i in range(n_layers):
    b = tf.keras.layers.Dense(n_units, activation='tanh')(b)
  out = tf.keras.layers.Dense(1, activation='linear')(b)

  out = tf.keras.layers.Lambda(lambda x: u0(x[1], x[2])*(1-(x[0]-t[0])/(t[1]-t[0])) + (x[0]-t[0])/(t[1]-t[0])*x[3])([inp1, inp2, inp3, out])

  model = tf.keras.models.Model([inp1, inp2, inp3], out)
  model.summary()

  return model

def build_model_soft_txy(t, x, y, n_layers, n_units):
  # Define the network
  inp1 = tf.keras.layers.Input(shape=(1,))
  b1 = Normalize(t[0], t[1])(inp1)

  inp2 = tf.keras.layers.Input(shape=(1,))
  b2 = Normalize(x[0], y[1])(inp2)

  inp3 = tf.keras.layers.Input(shape=(1,))
  b3 = Normalize(y[0], y[1])(inp3)

  b = tf.keras.layers.Concatenate()([b1, b2, b3])

  for i in range(n_layers):
    b = tf.keras.layers.Dense(n_units, activation='tanh')(b)
  out = tf.keras.layers.Dense(1, activation='linear')(b)

  model = tf.keras.models.Model([inp1, inp2, inp3], out)
  model.summary()

  return model