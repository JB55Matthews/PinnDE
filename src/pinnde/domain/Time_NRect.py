from .timedomain import timedomain
import numpy as np
import tensorflow as tf
from pyDOE import lhs

class Time_NRect(timedomain):

  def __init__(self, dim, xmins, xmaxs, timeRange):
      super().__init__(dim, timeRange)
      super().set_bdry_components(self._dim * 2)
      super().set_max_dim_vals(xmaxs)
      super().set_min_dim_vals(xmins)
      self._xmins = xmins
      self._xmaxs = xmaxs

  def isInside(self, timepoint):
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
    if (timepoint[0] == self._timeRange[0]):
      return True
    return False

  def sampleDomain(self, n_clp):
      points = lhs(self._dim, n_clp)
      for i in range (self._dim):
        points[:, i] = self._xmins[i] + (self._xmaxs[i] - self._xmins[i])*points[:, i]

      time_points = self._timeRange[0] + (self._timeRange[1] - self._timeRange[0])*lhs(1, n_clp).astype(np.float32)
      points = np.column_stack((time_points, points))
      return points
