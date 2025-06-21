import numpy as np
import tensorflow as tf
from pyDOE import lhs

class initials():
  """
  Class implementing initial conditions.
  """

  def __init__(self, domain, lambdas):
    """
    Constructor for class

    Args:
        domain (domain): Domain for boundary to act on.
        lambdas (list): List of initial functions as lambda functions, each function in full
          spatial dimensions, e.g, a 1+2 equation would have [lambda x1, x2: function].
    """
    self._domain = domain
    self._lambdas = lambdas
    self._orders = []
    self._init_block_size = None

  def get_domain(self):
    """
    Returns:
      (domain): Domain for boundary
    """
    return self._domain

  def get_lambdas(self):
    """
    Returns: 
      (list): Initial lambda functions
    """
    return self._lambdas

  def get_init_block_size(self):
    """
    Returns: 
      (int): Block size of each initial component
    """
    return self._init_block_size
  
  def get_orders(self):
    """
    Returns: 
      (list): Orders of t in each equation sampled
    """
    return self._orders

  def set_init_block_size(self, block_size):
    """
    Args: 
      block_size (int): Block size of each initial component
    """
    self._init_block_size = block_size


  def sampleInitials(self, n_iv):
    """
      Samples initial of domain, and computes initial functions to generate boundary data.

      Args:
        n_iv (int): Number of initial condition points to use.

      Returns:
        (tensor): Initial points.
      """
    flat_lambdas = self._lambdas
    self._orders = [len(self._lambdas)]
    # multiple eqns, [[lambdas], [lambdas]]
    if (type(self._lambdas[0]) == list):
      flat_lambdas = []
      self._orders = []
      for funcs in self._lambdas:
        i = 0
        for func in funcs:
          flat_lambdas.append(func)
          i += 1
        self._orders.append(i)
    
      # flat_lambdas = [x for xs in self._lambdas for x in xs]
  
    
    points = self._domain.sampleDomain(n_iv)
    func_points = []
    cols = np.shape(points)[1]
    for i in range(cols-1):
      func_points.append(points[:, i+1])
    for func in flat_lambdas:
      next_points = func(*func_points)
      points = np.column_stack([points, next_points])

    points[:, 0] = self._domain.get_timeRange()[0]
    # print(points)
    # time_points = self._domain.get_timeRange()[0] + 0*lhs(1, n_iv).astype(np.float32)
    # points = np.column_stack((time_points, points))
    return points