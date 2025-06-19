from .domain import domain
import numpy as np
import tensorflow as tf
from pyDOE import lhs

class NRect(domain):
    """
    Class for solving purely spatial problems on N dimensional hyperrectangles
    """

    def __init__(self, dim, xmins, xmaxs):
      """
      Constructor for class

      Args:
          dim (int): Spatial dimension of domain.
          xmins (list): Minimum values along each dimension, e.g, [-1, -1].
          xmaxs (list): Maximum values along each dimension, e.g, [1, 1].
      """
      super().__init__(dim)
      super().set_bdry_components(self._dim * 2)
      super().set_max_dim_vals(xmaxs)
      super().set_min_dim_vals(xmins)
      self._xmins = xmins
      self._xmaxs = xmaxs

    def isInside(self, point):
      """
      Args:
          point (list): Point in spatial dimensions of the hyperrectangle

      Returns:
          (bool): True if point is interior to the hyperrectangle, False otherwise
      """
      for i in range(self._dim):
        if ((self._xmins[i] > point[i]) or (self._xmaxs[i] < point[i])):
          return False
      return True

    def onBoundary(self, point):
      """
      Args:
          point (list): Point in spatial dimensions of the hyperrectangle

      Returns:
          (bool): True if point is on the boundary of the hyperrectangle, False otherwise
      """
      for i in range(self._dim):
        # maybe replace with isclose to
        if ((self._xmins[i] == point[i]) or (self._xmaxs[i] == point[i])):
          return True
      return False

    def sampleBoundary(self, n_bc):
      """
      Samples boundary of hyperrectangle.

      Args:
          n_bc (int): Number of points to sample in the boundary.

      Returns:
          (tensor): Sampled boundary points.
      """
      sample_blocks = self._dim * 2
      sample_sizes = round(n_bc / sample_blocks)
      super().set_bdry_component_size(sample_sizes)
      super().set_bdry_components(self._dim * 2)
      min_points = lhs(self._dim, sample_sizes)
      min_points[:, 0] = 0
      max_points = lhs(self._dim, sample_sizes)
      max_points[:, 0] = 1
      points = np.concatenate((min_points, max_points), axis=0)
      for i in range(self._dim-1):
        min_points = lhs(self._dim, sample_sizes)
        min_points[:, i+1] = 0
        max_points = lhs(self._dim, sample_sizes)
        max_points[:, i+1] = 1
        points = np.concatenate((points, min_points), axis=0)
        points = np.concatenate((points, max_points), axis=0)
      for i in range (self._dim):
        points[:, i] = self._xmins[i] + (self._xmaxs[i] - self._xmins[i])*points[:, i]
      return points

    def sampleDomain(self, n_clp):
      """
      Samples interior of hyperrectangle.
      
      Args:
          n_clp (int): Number of points to sample in the interior of the ellipsoid.

      Returns:
          (tensor): Sampled interior points.
      """
      points = lhs(self._dim, n_clp)
      for i in range (self._dim):
        points[:, i] = self._xmins[i] + (self._xmaxs[i] - self._xmins[i])*points[:, i]
      return points
