from .boundaries import boundaries
from ..domain.timedomain import timedomain
import inspect
import numpy as np
import tensorflow as tf

class odeicbc(boundaries):

    def __init__(self, domain, conditions, flag="ic"):
        if (flag != "ic" and flag != "bc"):
            raise ValueError("flag must be ic or bc - initial conditons for ode or boundary conditions for ode")
      
        super().__init__(domain, 4)
        self._conditions = conditions
        self._flag = "ic"
        
    def get_conditions(self):
        return self._conditions
    
    def get_flag(self):
        return self._flag

    def boundaryPoints(self, n_bc):
        pure_boundary = self._domain.sampleBoundary(n_bc)
        if self._flag == "ic":
            pure_boundary[:] = pure_boundary[0][0]
        
        print(pure_boundary)
    
        return