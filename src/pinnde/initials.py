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
    flat_lambdas = self._lambdas
    
    # multiple eqns, [[lambdas], [lambdas]]
    if (type(self._lambdas[0]) == list):
      flat_lambdas = [x for xs in self._lambdas for x in xs]

    
    points = self._domain.sampleDomain(n_iv)
    func_points = []
    cols = np.shape(points)[1]
    for i in range(cols-1):
      func_points.append(points[:, i+1])
    for func in flat_lambdas:
      next_points = func(*func_points)
      points = np.column_stack([points, next_points])

    points[:, 0] = 0
    # print(points)
    # time_points = self._domain.get_timeRange()[0] + 0*lhs(1, n_iv).astype(np.float32)
    # points = np.column_stack((time_points, points))
    return points