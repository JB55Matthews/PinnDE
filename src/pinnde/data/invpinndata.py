from .data import data
import numpy as np

class invpinndata(data):
    """
    Class for data on purely spatial inverse problems with a pinn.
    """

    def __init__(self, domain, boundaries, dimdata, udata, n_clp=10000, n_bc=600):
        """
        Constructor for class

        Args:
          domain (domain): Domain to generate data on.
          boundaries (boundaries): Boundary to generate data on.
          dimdata (list): List of data across spatial dimensions in which udata is recorded on for each u,
            each as (N,) shape tensors.
          udata (list): List containing data for each u solving for to solve inverse cosntants, each as
            (N,) shape tensors
          n_clp (int): Number of collocation points.
          n_bc (int): Number of boundary condition points.
        """
        super().__init__(5)
        self._domain = domain
        self._boundaries = boundaries
        self._n_clp = n_clp
        self._n_bc = n_bc
        self._clp = domain.sampleDomain(n_clp)
        self._bcp = boundaries.boundaryPoints(n_bc)
        self._udata = udata
        self._dimdata = dimdata
        self._n_invp = np.shape(udata[0])[0]
        self._invp = self.makeInverseData(dimdata, udata)
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
    
    def get_n_invp(self):
      """
      Returns:
        (int): Number of data points for inverse data given.
      """
      return self._n_invp
    
    def get_invp(self):
      """
      Returns:
        (tensor): Sampled data points for inverse data.
      """
      return self._invp 
    
    def makeInverseData(self, dimdata, udata):
      """
      Combines data user provided into consistent trainable set

      Args:
        dimdata (list): List of data across spatial dimensions in which udata is recorded on for each u,
          each as (N,) shape tensors.
        udata (list): List containing data for each u solving for to solve inverse cosntants, each as
          (N,) shape tensors
      """
      ddata = np.column_stack(dimdata)
      funcdata = np.column_stack(udata)
      data = np.column_stack([ddata, funcdata])
      return data