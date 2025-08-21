from .dondata import dondata
import numpy as np
import tensorflow as tf
from ..boundaries import odeicbc

class timedondata(dondata):
  """
  Class for data on spato-temporal problems with a deep operator network.
  """

  def __init__(self, domain, boundaries, initials,
                 n_clp=10000, n_bc=600, n_ic=600, n_sensors=1000):
      """
      Constructor for class

      Args:
        domain (domain): Domain to generate data on.
        boundaries (boundaries): Boundary to generate data on.
        initials (initials): Initial conditions to generate data on.
        n_clp (int): Number of collocation points.
        n_bc (int): Number of boundary condition points.
        n_ic (int): Number of initial condition points.
        n_sensors (int): Number of sensors to sample u with.
      """

      self._initials = initials
      self._n_iv = n_ic
      self._icp = initials.sampleInitials(n_ic)
      super().__init__(domain, boundaries, n_clp, n_bc, n_sensors)
      # self._sensors = self.generate_sensors()
      self.set_data_type(4)

  def generate_sensors(self):
        """
        Function to generate sensors for network.

        Returns:
          (tensor): Sampled sensors.
        """
        if isinstance(self._boundaries, odeicbc):
            u = np.random.uniform(min(self._boundaries.get_conditions())-1, max(self._boundaries.get_conditions())+1, (self._n_clp,self._n_sensors))
            usensors = np.float32(u)
            return usensors
      
        else:
            # initfunc = self._initials.get_lambdas()[0]
            # scalars = np.linspace(-2, 2, self._n_clp)
            # # scalars = np.random.uniform(-2, 2, self._n_clp)
            # clps = self._domain.sampleDomain(self._n_sensors)
            # func_points = []
            # cols = np.shape(clps)[1]
            # for i in range(cols-1):
            #   func_points.append(clps[:, i+1])
            # u = initfunc(*func_points)

            # sensors = tf.tile(tf.expand_dims(u, 0), [self._n_clp, 1])
            # sensors = scalars[:, None] * sensors 
            # return sensors
            max_waves = 3
            clps = self._domain.sampleDomain(self._n_sensors)

            amplitudes = np.random.randn(self._n_clp, max_waves)
            phases = -np.pi*np.random.rand(self._n_clp, max_waves) + np.pi/2

            
            u = 0.0*np.repeat(np.expand_dims(clps[:,0:1].flatten(), axis=0), repeats=self._n_clp, axis=0)
            for i in range(max_waves):
              u += amplitudes[:,i:i+1]
              for i in range(self._domain.get_dim()):
                  u = u*tf.sin((i+1)*np.repeat(np.expand_dims(clps[:, i:i+1].flatten(), axis=0), repeats=self._n_clp, axis=0) + phases[:,i:i+1])
            usensors = np.float32(u.numpy())
            return usensors
        
  def get_icp(self):
    """
    Returns:
      (tensor): Sampled initial condition points.
    """
    return self._icp

  def get_initials(self):
    """
    Returns:
      (initials): Initial conditions data is generated on.
    """
    return self._initials
  
  def get_n_ic(self):
    """
    Returns:
      (int): Number of initial points sampled.
    """
    return self._n_iv