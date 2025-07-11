from .boundaries import boundaries
import numpy as np
import tensorflow as tf
import inspect

class neumann(boundaries):
    """
    Class implementing neumann boundary conditions.
    """
    #lambdas len = 1, all boundary, if not, have to have every boundary coverd
    def __init__(self, domain, lambdas):
      """
      Constructor for class

      Args:
        domain (domain): Domain for boundary to act on.
        lambdas (list): List of boundary functions as lambda functions, each function in full time and
          spatial dimension, e.g, a 1+2 equation would have [lambda t, x1, x2: function].
      """
      super().__init__(domain, 3)
      self._lambdas = lambdas

      for func in lambdas:
        args = (inspect.getfullargspec(func))[0]
        if (len(args) != domain.get_dim()) and (len(args) != domain.get_dim()+1):
            raise ValueError("Lambda functions for boundaries must be functions of time+spatial dimensions, even if variables not used. \
                            Examples; lambda t, x1: 0+0*x1+0*t, or lambda x1, x2, x3: 2*x2")

      if (len(lambdas) == 1):
        new_lambdas = []
        for i in range(domain.get_bdry_components()):
          new_lambdas.append(lambdas[0])
        self._lambdas = new_lambdas

    def get_lambdas(self):
      """
      Returns:
        (list): Boundary lambda functions
      """
      return self._lambdas

    def boundaryPoints(self, n_bc):
      """
      Samples boundary of domain, and computes boundary functions to generate boundary data.

      Args:
        n_bc (int): Number of boundary conditions to use.

      Returns:
        (tensor): Boundary points.
      """
      sampled_boundary = self._domain.sampleBoundary(n_bc)

      # if isinstance(self._domain, timedomain):
      #   time_points = sampled_boundary[:,0]
      #   pure_boundary = sampled_boundary[:,1:,]
      # else:
      #   pure_boundary = sampled_boundary

      pure_boundary = sampled_boundary

      func_inputs = []
      comp_boundary = pure_boundary.reshape(-1, self._domain.get_bdry_component_size(), pure_boundary.shape[1])
      for comp in comp_boundary:
        components = []
        for dim in range(np.shape(comp)[1]):
          components.append(comp[:, dim])
        func_inputs.append(components)
      out = []

      for i in range(self._domain.get_bdry_components()):
        out.append(self._lambdas[i](*func_inputs[i]))

      boundary_points = np.column_stack([pure_boundary, np.array(out).flatten()])

      return boundary_points  