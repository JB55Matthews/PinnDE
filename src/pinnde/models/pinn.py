from ..selectors import pinnSelectors, constraintSelector
from ..data import timepinndata
from ..training import pinnTrainSteps 
import tensorflow as tf
import numpy as np

class pinn():
    """
    Class implementing a pinn architecture
    """

    def __init__(self, data, eqns,
                 layers=4, units=60, inner_act="tanh",
                 out_act="linear", constraint="soft"):
        
        """
        Constructor for class.

        Args:
          data (data): Data to solve on pinn.
          eqns (list): List of eqns to solve as strings. Spatial dimensions x1, x2, etc. Time dimension t. 
            If a single equation, use u, if multiple, u1, u2, etc. See tutorials how to do specific examples.
          layers (int): Number of internal layers for pinn.
          units (int): Number of units for internal layers for pinn.
          inner_act (string): Activation function for internal layers of pinn. Must be TensorFlow useable activation function.
          out_act (string): Activation function for internal layers of pinn. Must be TensorFlow useable activation function.
          constraint (string): Soft or hard constraint for network. **Note only soft currently work, for hard, use legacy models.**
        """

        self._data = data
        self._eqns = eqns
        self._domain = data.get_domain()
        self._boundaries = data.get_boundaries()
        self._clp = data.get_clp()
        self._bcp = data.get_bcp()
        self._dim = self._domain.get_dim()
        self._bdry_type = self._boundaries.get_bdry_type()
        self._layers = layers
        self._units = units
        self._inner_act = inner_act
        self._out_act = out_act
        self._constraint = constraint

        n = self._domain.get_dim()
        pt_maxes = self._clp.max(axis=0)
        pt_mins = self._clp.min(axis=0)

        self._epochs = None
        self._epoch_loss = None
        self._clp_loss = None
        self._bc_loss = None
        self._ic_loss = None
        self._eqns = eqns

        if isinstance(data, timepinndata):
          self._initials = data.get_initials()
          self._icp = data.get_icp()
          self._t_orders = self._initials.get_orders()
          n += 1
        else:
          self._initials = None

        # ---------
        inlist = []
        blist = []

        for i in range(n):
            input = tf.keras.layers.Input(shape=(1,))
            # time commponent always is normalized even in periodic
            if isinstance(data, timepinndata) and (i == 0):
              bin = Normalize(pt_mins[i], pt_maxes[i])(input)
            elif (self._boundaries.get_bdry_type() == 1):
              bin = Periodic(pt_mins[i], pt_maxes[i])(input)
            else:
              bin = Normalize(pt_mins[i], pt_maxes[i])(input)

            inlist.append(input)
            blist.append(bin)

        b = tf.keras.layers.Concatenate()(blist)

        for i in range(layers):
            b = tf.keras.layers.Dense(units, activation=inner_act)(b)

        outs = []
        for i in range(len(eqns)):
          out = tf.keras.layers.Dense(1, activation=out_act)(b)
          outs.append(out)

        if pinnSelectors.pinnSelector(self._constraint)():
          if constraintSelector.constraintSelector(self._domain, self._boundaries, self._eqns, self._initials) != None:
            out = tf.keras.layers.Lambda(constraintSelector.constraintSelector(self._domain, self._boundaries, self._eqns, self._initials), 
                                         output_shape=(1,))([inlist, out])
          pass

        model = tf.keras.models.Model(inlist, outs)
        model.summary()

        self._network = model
# --------------------------------

    def get_network(self):
      """
      Returns:
        (model): TensorFlow model.
      """
      return self._network
    
    def get_epoch_loss(self):
      """
      Returns:
        (tensor): Epoch loss after training.
      """
      return self._epoch_loss
    
    def get_domain(self):
      """
      Returns:
        (domain): Domain pinn is trained on.
      """
      return self._domain
    
    def get_epochs(self):
      """
      Returns:
        (int): Epochs pinn was trained for.
      """
      return self._epochs

    def get_boundaries(self):
      """
      Returns:
        (boundaries): Boundaries pinn was trained on.
      """
      return self._boundaries
    
    def get_eqns(self):
      """
      Returns:
        (list): Equations pinn was trained for.
      """
      return self._eqns
  

    def train(self, epochs, opt="adam", meta="false", adapt_pt="false"):
      """
      Main training function
      
      Args:
        epochs (int): Epochs to train for.
        opt (string): Optimizer to use.
        meta (string): Whether to meta-learned optimize. **Not implemented**.
        adapt_pt (string): Adaptive point sampling strategy to use. **Not implemented**.
      """
      self._epochs = epochs
      if isinstance(self._data, timepinndata):
        self.trainTime(self._eqns, epochs, opt, meta, adapt_pt)
      else:
        self.trainNoTime(self._eqns, epochs, opt, meta, adapt_pt)      
    

    def trainNoTime(self, eqns, epochs, opt, meta, adapt_pt):
      """
      Main setup for training and training loop which calls trainStep. This is used for a purely spatial problem

      Args:
        eqns (list): List of eqns to solve as strings.  
        epochs (int): Epochs to train for.
        opt (string): Optimizer to use.
        meta (string): Whether to meta-learned optimize. **Not implemented**.
        adapt_pt (string): Adaptive point sampling strategy to use. **Not implemented**.
      """

      lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, epochs, 1e-4)
      opt = pinnSelectors.pinnSelector(opt)(lr)
      bs_clp, bs_bcp = self._data.get_n_clp(), self._data.get_n_bc()

      ds_clp = tf.data.Dataset.from_tensor_slices(self._data.get_clp())
      ds_clp = ds_clp.cache().shuffle(self._data.get_n_clp()).batch(bs_clp)

      ds_bc = tf.data.Dataset.from_tensor_slices(self._data.get_bcp())
      ds_bc = ds_bc.cache().shuffle(self._data.get_n_bc()).batch(bs_bcp)

      ds = tf.data.Dataset.zip((ds_clp, ds_bc))
      ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

      epoch_loss = np.zeros(epochs)
      clp_loss = np.zeros(epochs)
      bc_loss = np.zeros(epochs)

      BCloss = None
      CLPloss = None

      for i in range(epochs):

        n_batches = 0

        for (clps, bcs) in ds:

          CLPloss, BCloss, grads = pinnTrainSteps.trainStep(eqns, clps, bcs, self._network, self._boundaries, self._constraint)
          opt.apply_gradients(zip(grads, self._network.trainable_variables))
          n_batches += 1
          epoch_loss[i] += CLPloss + BCloss
          clp_loss[i] += CLPloss
          bc_loss[i] += BCloss

        epoch_loss[i] /= n_batches

        if (np.mod(i, 100)==0):
            print("CLP loss, BC loss in {}th epoch: {: 6.4f}, {: 6.4f}.".format(i, CLPloss.numpy(), BCloss.numpy()))

      self._epoch_loss = epoch_loss
      self._clp_loss = clp_loss
      self._bc_loss = bc_loss
      self._eqns = eqns
      return

    def trainTime(self, eqns, epochs, opt, meta, adapt_pt):
      """
      Main setup for training and training loop which calls trainStep. This is used for a spatio-temporal problem

      Args:
        eqns (list): List of eqns to solve as strings.  
        epochs (int): Epochs to train for.
        opt (string): Optimizer to use.
        meta (string): Whether to meta-learned optimize. **Not implemented**.
        adapt_pt (string): Adaptive point sampling strategy to use. **Not implemented**.
      """

      lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, epochs, 1e-4)
      opt = pinnSelectors.pinnSelector(opt)(lr)
      bs_clp, bs_bcp, bs_icp = self._data.get_n_clp(), self._data.get_n_bc(), self._data.get_n_ic()

      ds_clp = tf.data.Dataset.from_tensor_slices(self._data.get_clp())
      ds_clp = ds_clp.cache().shuffle(self._data.get_n_clp()).batch(bs_clp)

      ds_bc = tf.data.Dataset.from_tensor_slices(self._data.get_bcp())
      ds_bc = ds_bc.cache().shuffle(self._data.get_n_bc()).batch(bs_bcp)

      ds_ic = tf.data.Dataset.from_tensor_slices(self._data.get_icp())
      ds_ic = ds_ic.cache().shuffle(self._data.get_n_ic()).batch(bs_icp)

      ds = tf.data.Dataset.zip((ds_clp, ds_bc, ds_ic))
      ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

      epoch_loss = np.zeros(epochs)
      clp_loss = np.zeros(epochs)
      bc_loss = np.zeros(epochs)
      ic_loss = np.zeros(epochs)

      for i in range(epochs):

        n_batches = 0

        for (clps, bcs, ics) in ds:
          
          CLPloss, BCloss, ICloss, grads = pinnTrainSteps.trainStepTime(eqns, clps, bcs, ics, self._network, 
                                                                      self._boundaries, self._t_orders, self._constraint)
          opt.apply_gradients(zip(grads, self._network.trainable_variables))
          n_batches += 1
          epoch_loss[i] += CLPloss + BCloss
          clp_loss[i] += CLPloss
          bc_loss[i] += BCloss
          ic_loss[i] += ICloss

        epoch_loss[i] /= n_batches

        if (np.mod(i, 100)==0):
            print("CLP loss, BC loss, IV loss in {}th epoch: {: 6.4f}, {: 6.4f}, {: 6.4f}.".format(i, CLPloss.numpy(), BCloss.numpy(), ICloss.numpy()))

      self._epoch_loss = epoch_loss
      self._clp_loss = clp_loss
      self._bc_loss = bc_loss
      self._ic_loss = ic_loss
      self._eqns = eqns
      return

    

# Define the normalization layer
class Normalize(tf.keras.layers.Layer):
  """
  Class which describes a normalize layer for pinn. Returns input data
  normalized to interval [-1, 1].
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
  Class which describes a Periodic layer for pinn. Used in periodic models.
  """
  def __init__(self, xmin, xmax):
    super(Periodic, self).__init__()
    self.xmin = xmin
    self.xmax = xmax

  def call(self, inputs):
    return tf.concat((tf.cos(2*np.pi*(inputs/(self.xmax - self.xmin))), tf.sin(2*np.pi*(inputs/(self.xmax - self.xmin)))), axis=1)