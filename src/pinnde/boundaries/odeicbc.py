from .boundaries import boundaries
from ..initials import initials
import inspect
import numpy as np
import tensorflow as tf

class odeicbc(boundaries):

    def __init__(self, domain, conditions, flag="ic"):
        # ic: conditions [u0, ux10, ..]
        # bc: conditions [u0, uf, ux10, ux1f, ...]
        if (flag != "ic" and flag != "bc"):
            raise ValueError("flag must be ic or bc - initial conditons for ode or boundary conditions for ode")
      
        super().__init__(domain, 4)
        self._conditions = conditions
        self._flag = flag
        self._orders = None
        
    def get_conditions(self):
        return self._conditions
    
    def get_flag(self):
        return self._flag
    
    def get_orders(self):
        return self._orders

    def boundaryPoints(self, n_bc):
        if self._flag == "ic":
            flat_conds = self._conditions
            self._orders = [len(self._conditions)]
            # multiple eqns, [[lambdas], [lambdas]]
            if (type(self._conditions[0]) == list):
                flat_conds = []
                self._orders = []
                for conds in self._conditions:
                    i = 0
                    for cond in conds:
                        flat_conds.append(cond)
                        i += 1
                    self._orders.append(i)
            
            points = self._domain.sampleBoundary(n_bc)
            points[:] = points[0][0]
            pts_shape = points
            for cond in flat_conds:
                next_points = cond+0*pts_shape
                points = np.column_stack([points, next_points])
        
        return points