from .domain import domain
import numpy as np
import tensorflow as tf
from pyDOE import lhs

class NEllipsoid(domain):
    """
    Class for solving purely spatial problems on N dimensional ellipsoids
    """

    def __init__(self, dim, center, semilengths):
        """
        Constructor for class

        Args:
            dim (int): Spatial dimension of domain.
            center (list): Center of ellipsoid in list for, e.g, [0, 0].
            semilengths (list): semi-axi lengths of ellipsoid, e.g, [1, 1].
        """
        if dim == 1:
            raise ValueError("1+1 equation should use a Time_NRect, as 1 spatial dimension is an interval")
        
        super().__init__(dim)
        super().set_bdry_components(1)
        self._center = center
        self._semilengths = semilengths
        max_dims = []
        min_dims = []
        for i in range(dim):
            max_dims.append(center[i] + semilengths[i])
            min_dims.append(center[i] - semilengths[i])
        super().set_max_dim_vals(max_dims)
        super().set_min_dim_vals(min_dims)

    def isInside(self, point):
        """
        Args:
            point (list): Point in spatial dimensions of the ellipsoid

        Returns:
            (bool): True if point is interior to the ellipsoid, False otherwise
        """
        if (len(point) == self._dim):
            sum = 0
            for i in range(self._dim):
                sum += ((point[i]-self._center[i])**2)/(self._semilengths[i]**2)
            if sum < 1:
                return True
            return False

    def onBoundary(self, point):
        """
        Args:
            point (list): Point in spatial dimensions of the ellipsoid

        Returns:
            (bool): True if point is on the boundary of the ellipsoid, False otherwise
        """
        if (len(point) == self._dim):
            sum = 0
            for i in range(self._dim):
                sum += ((point[i]-self._center[i])**2)/(self._semilengths[i]**2)
            if np.isclose(sum, 1, 0.01):
                return True
            return False

    def sampleBoundary(self, n_bc):
        """
        Samples boundary of ellipsoid

        Args:
            n_bc (int): Number of points to sample in the boundary.

        Returns:
            (tensor): Sampled boundary points.
        """
        super().set_bdry_component_size(n_bc)
        points = []
        for i in range(self._dim):
            points.append(0)
        points = np.array([points])
        while np.shape(points)[0] < n_bc+1:
        
            new_point = lhs(self._dim, 1)
            for i in range (self._dim):
                new_point[:, i] = self.get_min_dim_vals()[i] + (self.get_max_dim_vals()[i] - self.get_min_dim_vals()[i])*new_point[:, i]
            if self.onBoundary(new_point.flatten()):
                points = np.concatenate((points, new_point), axis=0)

        points = np.delete(points, (0), axis=0)
        return points

    def sampleDomain(self, n_clp):
        """
        Samples interior of ellipsoid
        
        Args:
            n_clp (int): Number of points to sample in the interior of the ellipsoid.

        Returns:
            (tensor): Sampled interior points.
        """
        points = []
        for i in range(self._dim):
            points.append(0)
        points = np.array([points])
        while np.shape(points)[0] < n_clp+1:
            new_point = lhs(self._dim, 1)
            for i in range (self._dim):
                new_point[:, i] = self.get_min_dim_vals()[i] + (self.get_max_dim_vals()[i] - self.get_min_dim_vals()[i])*new_point[:, i]
            if self.isInside(new_point.flatten()):
                points = np.concatenate((points, new_point), axis=0)

        points = np.delete(points, (0), axis=0)
        return points