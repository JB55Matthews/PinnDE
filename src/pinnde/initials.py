import numpy as np
import tensorflow as tf
from pyDOE import lhs

class initials():

  def __init__(self, domain, lambdas):
    self._domain = domain
    self._lambdas = lambdas
    self._init_block_size = None

  def get_domain(self):
    return self._domain

  def get_lambdas(self):
    return self._lambdas

  def get_init_block_size(self):
    return self._init_block_size

  def set_init_block_size(self, block_size):
    self._init_block_size = block_size

  def sampleInitials(self, n_iv):
    points = self._domain.sampleDomain(n_iv)
    func_points = []
    cols = points.ndim
    for i in range(cols):
      func_points.append(points[:, i+1])
    for func in self._lambdas:
      next_points = func(*func_points)
      points = np.column_stack([points, next_points])

    points[:, 0] = 0
    # time_points = self._domain.get_timeRange()[0] + 0*lhs(1, n_iv).astype(np.float32)
    # points = np.column_stack((time_points, points))
    return points