from .model import model
import tensorflow as tf
import numpy as np

class pinn(model):

    def __init__(self, data, 
                 layers=4, units=40, inner_act="tanh",
                 out_act="linear", hard_constraint="false"):
        
        self._domain = data.get_domain()
        self._boundaries = data.get_boundaries()
        self._initials = data.get_initials()
        self._layers = layers
        self._units = units
        self._inner_act = inner_act
        self._out_act = out_act
        self._hard_constraint = hard_constraint

        n = 3
        t_bdry = [0,1]

        inlist = []
        blist = []

        for i in range(n):
            input = tf.keras.layers.Input(shape=(1,))
            bin = Periodic(t_bdry[0], t_bdry[1])(input)
            inlist.append(input)
            blist.append(bin)

        b = tf.keras.layers.Concatenate()(blist)

        for i in range(layers):
            b = tf.keras.layers.Dense(units, activation=inner_act)(b)
        out = tf.keras.layers.Dense(1, activation=out_act)(b)

        model = tf.keras.models.Model(inlist, out)
        model.summary()

        self._network = model
    
    def train(self, eqns, epochs, opt="adam", meta="false", adapt_pt="false"):
        return
    

# Define the normalization layer
class Normalize(tf.keras.layers.Layer):
  """
  Class which describes a normalize layer for PINN. Returns input data
  normalized to interval [-1, 1].

  Models for solving solvePDE_tx equations
  --------------------------------------
  """
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
  
class Periodic(tf.keras.layers.Layer):
  """
  Class which describes a Periodic layer for PINN. Used in periodic models
  """
  def __init__(self, xmin, xmax):
    super(Periodic, self).__init__()
    self.xmin = xmin
    self.xmax = xmax

  def call(self, inputs):
    return tf.concat((tf.cos(2*np.pi*(inputs/(self.xmax - self.xmin))), tf.sin(2*np.pi*(inputs/(self.xmax - self.xmin)))), axis=1)