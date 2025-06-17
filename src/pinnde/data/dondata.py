from .data import data
import numpy as np
import tensorflow as tf
from ..domain import timedomain, domain


class dondata(data):

    def __init__(self, domain, boundaries, n_clp=1000, n_bc=100, n_sensors=100):
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
      return self._clp

    def get_bcp(self):
      return self._bcp

    def get_domain(self):
        return self._domain

    def get_boundaries(self):
        return self._boundaries
    
    def get_n_clp(self):
      return self._n_clp

    def get_n_bc(self):
      return self._n_bc