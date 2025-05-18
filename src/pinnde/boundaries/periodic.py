from .boundaries import boundaries
import numpy as np

class periodic(boundaries):

    def __init__(self, domain):
        super().__init__(domain, 1)
        return
    
    def boundaryPoints(self, n_bc):
        sampled_boundary = self._domain.sampleBoundary(n_bc)
        sampled_boundary[:] = 0
        return sampled_boundary