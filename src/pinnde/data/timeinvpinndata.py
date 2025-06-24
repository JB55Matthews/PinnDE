from .pinndata import pinndata
import numpy as np

class timeinvpinndata(pinndata):
  """
  Class for data on spato-temporal inverse problems with a pinn.
  """

  def __init__(self, domain, boundaries, initials, dimdata, udata,
                 n_clp=10000, n_bc=600, n_ic=600):
      """
      Constructor for class

      Args:
        domain (domain): Domain to generate data on.
        boundaries (boundaries): Boundary to generate data on.
        initials (initials): Initial conditions to generate data on.
        dimdata (list): List of data across time+spatial dimensions in which udata is recorded on for each u,
          each as (N,) shape tensors.
        udata (list): List containing data for each u solving for to solve inverse cosntants, each as
          (N,) shape tensors
        n_clp (int): Number of collocation points.
        n_bc (int): Number of boundary condition points.
        n_ic (int): Number of initial condition points.
      """

      super().__init__(domain, boundaries, n_clp, n_bc)
      self.set_data_type(6)
      self._initials = initials
      self._n_iv = n_ic
      self._icp = initials.sampleInitials(n_ic)
      self._udata = udata
      self._dimdata = dimdata
      self._n_invp = np.shape(udata[0])[0]
      self._invp = self.makeInverseData(dimdata, udata)

  def get_icp(self):
    """
    Returns:
      (tensor): Sampled initial condition points.
    """
    return self._icp

  def get_initials(self):
    """
    Returns:
      (initials): Initial conditions data is generated on.
    """
    return self._initials
  
  def get_n_ic(self):
    """
    Returns:
      (int): Number of initial points sampled.
    """
    return self._n_iv
  
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
      dimdata (list): List of data across time+spatial dimensions in which udata is recorded on for each u,
        each as (N,) shape tensors.
      udata (list): List containing data for each u solving for to solve inverse cosntants, each as
        (N,) shape tensors
    """
    ddata = np.column_stack(dimdata)
    funcdata = np.column_stack(udata)
    data = np.column_stack([ddata, funcdata])
    return data