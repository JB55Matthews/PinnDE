from abc import ABC, abstractmethod

class domain(ABC):

    def __init__(self, dim):
      # inherited classes must call set_max_dim_vals and set_min_dim_vals
      self._dim = dim
      self._bdry_component_size = None
      self._bdry_components = 1
      self._max_dim_vals = []
      self._min_dim_vals = []

    def get_dim(self):
      return self._dim

    def get_bdry_component_size(self):
      return self._bdry_component_size
    
    def get_max_dim_vals(self):
      return self._max_dim_vals
    
    def get_min_dim_vals(self):
      return self._min_dim_vals

    def set_bdry_component_size(self, component_block):
      self._bdry_component_size = component_block

    def get_bdry_components(self):
      return self._bdry_components

    def set_bdry_components(self, components):
      self._bdry_components = components

    def set_max_dim_vals(self, max_dim_vals):
      self._max_dim_vals = max_dim_vals

    def set_min_dim_vals(self, min_dim_vals):
      self._min_dim_vals = min_dim_vals


    @abstractmethod
    def isInside(self, point):
      pass

    @abstractmethod
    def onBoundary(self, point):
      pass

    @abstractmethod
    def sampleBoundary(self, n_bc):
      # must include time points when implementing time domain as first col
      # can implement ray casting for general api
      pass

    @abstractmethod
    def sampleDomain(self, n_clp):
      #
      pass