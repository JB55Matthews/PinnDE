from .boundaries import boundaries
from ..initials import initials
import inspect
import numpy as np
import tensorflow as tf

class odeicbc(boundaries):
    """
    Class implementing ode initial and boundary conditions.
    """

    def __init__(self, domain, conditions, flag="ic"):
        """
        Constructor for class

        Args:
            domain (domain): Domain for boundary to act on. Must time NRect with 1 dimension
            conditions (list): List of initial or boundary values of equations.
            flag (string): ic or bc to determine what condition is being used.
                **Only ic implemented currently**.
        """
        # ic: conditions [u0, ux10, ..]
        # bc: conditions [u0, uf, ux10, ux1f, ...]
        if (flag != "ic" and flag != "bc"):
            raise ValueError("flag must be ic or bc - initial conditions for ode or boundary conditions for ode")
      
        super().__init__(domain, 4)
        self._conditions = conditions
        self._flag = flag
        self._orders = None
        
    def get_conditions(self):
        """
        Returns:
            (list): ic or bc conditions
        """
        return self._conditions
    
    def get_flag(self):
        """
        Returns:
            (string): flag, ic or bc
        """
        return self._flag
    
    def get_orders(self):
        """
        Returns:
            (list): order(s) of equations
        """
        return self._orders

    def boundaryPoints(self, n_bc):
        """
        Samples boundary of domain, and computes boundary functions to generate boundary data.

        Args:
            n_bc (int): Number of boundary conditions to use.

        Returns:
            (tensor): Boundary points.
        """
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