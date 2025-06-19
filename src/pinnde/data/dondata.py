from .data import data
import numpy as np
import tensorflow as tf
from ..domain import timedomain, domain


class dondata(data):
    """
    Class for data on purely spatial problems with a deep operator network.
    """

    def __init__(self, domain, boundaries, n_clp=10000, n_bc=600, n_sensors=1000):
        """
        Constructor for class

        Args:
          domain (domain): Domain to generate data on.
          boundaries (boundaries): Boundary to generate data on.
          n_clp (int): Number of collocation points.
          n_bc (int): Number of boundary condition points.
          n_sensors (int): Number of sensors to sample u with.
        """
        super().__init__(3)
        self._domain = domain
        self._boundaries = boundaries
        self._n_clp = n_clp
        self._n_bc = n_bc
        self._n_sensors = n_sensors
        self._clp = domain.sampleDomain(n_clp)
        self._bcp = boundaries.boundaryPoints(n_bc)
        self._sensors = self.generate_sensors()
        return

    def generate_sensors(self):
        """
        Function to generate sensors for network.

        Returns:
          (tensor): Sampled sensors.
        """
        max_waves = 3
        clps = self._domain.sampleDomain(self._n_sensors)
        if (isinstance(self._domain, timedomain)):
           clps = clps[:,1:]

        amplitudes = np.random.randn(self._n_clp, max_waves)
        phases = -np.pi*np.random.rand(self._n_clp, max_waves) + np.pi/2

        
        u = 0.0*np.repeat(np.expand_dims(clps[:,0:1].flatten(), axis=0), repeats=self._n_clp, axis=0)
        for i in range(max_waves):
           u += amplitudes[:,i:i+1]
           for i in range(self._domain.get_dim()):
              u = u*tf.sin((i+1)*np.repeat(np.expand_dims(clps[:, i:i+1].flatten(), axis=0), repeats=self._n_clp, axis=0) + phases[:,i:i+1])
        usensors = np.float32(u.numpy())
        return usensors

    def get_clp(self):
      """
      Returns:
        (tensor): Sampled collocation points.
      """
      return self._clp

    def get_bcp(self):
      """
      Returns:
        (tensor): Sampled boundary points.
      """
      return self._bcp

    def get_domain(self):
      """
      Returns:
        (domain): Domain data is generated on.
      """
      return self._domain

    def get_boundaries(self):
      """
      Returns:
        (boundaries): Boundary data is generated on.
      """
      return self._boundaries
    
    def get_n_clp(self):
      """
      Returns:
        (int): Number of collocation points sampled.
      """
      return self._n_clp

    def get_n_bc(self):
      """
      Returns:
        (int): Number of boundary points sampled.
      """
      return self._n_bc
    
    def get_n_sensors(self):
       """
      Returns:
        (int): Number of sensors sampled.
      """
       return self._n_sensors
    
    def get_sensors(self):
       """
      Returns:
        (tensor): Sampled sensors.
      """
       return self._sensors