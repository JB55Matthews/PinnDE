from .boundaries import boundaries
import numpy as np

class periodic(boundaries):
    """
    Class implementing periodic boundary conditions.
    """

    def __init__(self, domain):
        """
      Constructor for class

        Args:
            domain (domain): Domain for boundary to act on.
      """

        super().__init__(domain, 1)
        return
    
    def boundaryPoints(self, n_bc):
        """
      Samples boundary of domain, and computes zero-value boundary points as this is ignored in training.

      Args:
        n_bc (int): Number of boundary conditions to use.

      Returns:
        (tensor): Boundary points.
      """
        sampled_boundary = self._domain.sampleBoundary(n_bc)
        sampled_boundary[:] = 0
        return sampled_boundary