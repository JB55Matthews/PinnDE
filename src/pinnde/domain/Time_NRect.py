from .timedomain import timedomain
import numpy as np
import tensorflow as tf
from pyDOE import lhs

class Time_NRect(timedomain):
  """
  Class for solving purely spatial problems on N dimensional hyperrectangles
  """

  def __init__(self, dim, xmins, xmaxs, timeRange):
      """
      Constructor for class

      Args:
          dim (int): Spatial dimension of domain.
          xmins (list): Minimum values along each dimension, e.g, [-1, -1].
          xmaxs (list): Maximum values along each dimension, e.g, [1, 1].
          timeRange (list): Range of time to solve equation over, e.g, [0, 1].
      """
      super().__init__(dim, timeRange)
      super().set_bdry_components(self._dim * 2)
      super().set_max_dim_vals(xmaxs)
      super().set_min_dim_vals(xmins)
      self._xmins = xmins
      self._xmaxs = xmaxs

  def isInside(self, timepoint):
    """
    Args:
        timepoint (list): Point in time+spatial dimensions of the hyperrectangle

    Returns:
        (bool): True if point is interior to the hyperrectangle, False otherwise
    """
    # No time componennt
    if (len(timepoint) == self._dim):
      for i in range(self._dim):
        if ((self._xmins[i] > timepoint[i]) or (self._xmaxs[i] < timepoint[i])):
          return False
      return True
    # time components
    elif (len(timepoint) == self._dim + 1):
      for i in range(self._dim):
        if ((self._xmins[i] > timepoint[i+1]) or (self._xmaxs[i] < timepoint[i+1])):
          return False
      return True
    

  def onBoundary(self, timepoint):
    """
    Args:
        timepoint (list): Point in time+spatial dimensions of the hyperrectangle

    Returns:
        (bool): True if point is on the boundary of the hyperrectangle, False otherwise
    """
    # No time componennt
    if (len(timepoint) == self._dim):
      for i in range(self._dim):
        # maybe replace with isclose to
        if ((self._xmins[i] == timepoint[i]) or (self._xmaxs[i] == timepoint[i])):
          return True
      return False
    # time components
    elif (len(timepoint) == self._dim + 1):
      for i in range(self._dim):
        # maybe replace with isclose to
        if ((self._xmins[i] == timepoint[i+1]) or (self._xmaxs[i] == timepoint[i+1])):
          return True
      return False

  def sampleBoundary(self, n_bc):
      """
      Samples boundary of hyperrectangle.

      Args:
          n_bc (int): Number of points to sample in the boundary.

      Returns:
          (tensor): Sampled time+boundary points.
      """
      sample_blocks = self._dim * 2
      sample_sizes = round(n_bc / sample_blocks)
      super().set_bdry_component_size(sample_sizes)
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

      time_points = self._timeRange[0] + (self._timeRange[1] - self._timeRange[0])*lhs(1, 2*sample_sizes*self._dim).astype(np.float32)
      points = np.column_stack((time_points, points))
      return points

  def onInitial(self, timepoint):
    """
    Args:
      timepoint (list): Point in time+spatial dimensions of domain.

    Returns:
      (bool): True if point is an initial point, False otherwise.
    """
    if (timepoint[0] == self._timeRange[0]):
      return True
    return False

  def sampleDomain(self, n_clp):
      """
      Samples interior of hyperrectangle.
      
      Args:
          n_clp (int): Number of points to sample in the interior of the ellipsoid.

      Returns:
          (tensor): Sampled time+interior points.
      """
      points = lhs(self._dim, n_clp)
      for i in range (self._dim):
        points[:, i] = self._xmins[i] + (self._xmaxs[i] - self._xmins[i])*points[:, i]

      time_points = self._timeRange[0] + (self._timeRange[1] - self._timeRange[0])*lhs(1, n_clp).astype(np.float32)
      points = np.column_stack((time_points, points))
      return points
