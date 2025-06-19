from abc import ABC, abstractmethod

class domain(ABC):
    """
    Abstract class for domains
    """

    def __init__(self, dim):
      # inherited classes must call set_max_dim_vals and set_min_dim_vals
      """
      Constructor for class

      Args:
        dim (int): Spatial dimension of domain.

      """
      self._dim = dim
      self._bdry_component_size = None
      self._bdry_components = 1
      self._max_dim_vals = []
      self._min_dim_vals = []

    def get_dim(self):
      """
      Returns:
        (int): Spatial dimension.
      """
      return self._dim

    def get_bdry_component_size(self):
      """
      Returns:
        (int): Internal size of boundary components.
      """
      return self._bdry_component_size
    
    def get_max_dim_vals(self):
      """
      Returns:
        (list): Maximum vales along each dimension axis.
      """
      return self._max_dim_vals
    
    def get_min_dim_vals(self):
      """
      Returns:
        (list): Minimum vales along each dimension axis.
      """
      return self._min_dim_vals

    def set_bdry_component_size(self, component_block):
      """
      Args:
        component_block (int): Size of boundary component.
      """
      self._bdry_component_size = component_block

    def get_bdry_components(self):
      """
      Returns:
        (int): Number of boundary components.
      """
      return self._bdry_components

    def set_bdry_components(self, components):
      """
      Args:
        components (int): Number boundary components.
      """
      self._bdry_components = components

    def set_max_dim_vals(self, max_dim_vals):
      """
      Args:
        max_dim_vals (list): Maximum values along each axis of domain.
      """
      self._max_dim_vals = max_dim_vals

    def set_min_dim_vals(self, min_dim_vals):
      """
      Args:
        min_dim_vals (list): Minimum values along each axis of domain.
      """
      self._min_dim_vals = min_dim_vals


    @abstractmethod
    def isInside(self, point):
      """
      Abstract method all domains must specify. Determines whether a point is in the interior of a domain.

      Args:
        point (list): Point in spatial dimensions of domain

      Returns:
        (bool): True if point is interior to the domain, False otherwise
      """
      pass

    @abstractmethod
    def onBoundary(self, point):
      """
      Abstract method all domains must specify. Determines whether a point is on the boundary of a domain.

      Args:
        point (list): Point in spatial dimensions of domain

      Returns:
        (bool): True if point is interior on the boundary of the domain, False otherwise
      """
      pass

    @abstractmethod
    def sampleBoundary(self, n_bc):
      """
      Abstract method all domains must specify. Must specify how to sample the boundary of the domain. Must include time dimension
        in first column if a timedomain.

      Args:
        n_bc (int): Number of points to sample in the boundary.

      Returns sampled boundary points.
      """
      # must include time points when implementing time domain as first col
      # can implement ray casting for general api
      pass

    @abstractmethod
    def sampleDomain(self, n_clp):
      """
      Abstract method all domains must specify. Must specify how to sample the domain. Must include time dimension
        in first column if a timedomain.

      Args:
        n_clp (int): Number of points to sample in the domain.

      Returns sampled domain points.
      """
      #
      pass