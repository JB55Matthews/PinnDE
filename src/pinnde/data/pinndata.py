from .data import data

class pinndata(data):
    """
    Class for data on purely spatial problems with a pinn.
    """

    def __init__(self, domain, boundaries, n_clp=10000, n_bc=600):
        """
        Constructor for class

        Args:
          domain (domain): Domain to generate data on.
          boundaries (boundaries): Boundary to generate data on.
          n_clp (int): Number of collocation points.
          n_bc (int): Number of boundary condition points.
        """
        super().__init__(1)
        self._domain = domain
        self._boundaries = boundaries
        self._n_clp = n_clp
        self._n_bc = n_bc
        self._clp = domain.sampleDomain(n_clp)
        self._bcp = boundaries.boundaryPoints(n_bc)
        return

    def get_clp(self):
      """
      Returns:
        (tensor): Sampled collocation points.
      """
      return self._clp

    def get_bcp(self):
      """
      Returns:
        (tensor): Sampled boundary points.
      """
      return self._bcp

    def get_domain(self):
      """
      Returns:
        (domain): Domain data is generated on.
      """
      return self._domain

    def get_boundaries(self):
      """
      Returns:
        (boundaries): Boundary data is generated on.
      """
      return self._boundaries
    
    def get_n_clp(self):
      """
      Returns:
        (int): Number of collocation points sampled.
      """
      return self._n_clp

    def get_n_bc(self):
      """
      Returns:
        (int): Number of boundary points sampled.
      """
      return self._n_bc