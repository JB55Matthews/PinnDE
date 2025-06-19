from .dondata import dondata

class timedondata(dondata):
  """
  Class for data on spato-temporal problems with a deep operator network.
  """

  def __init__(self, domain, boundaries, initials,
                 n_clp=10000, n_bc=600, n_ic=600, n_sensors=1000):
      """
      Constructor for class

      Args:
        domain (domain): Domain to generate data on.
        boundaries (boundaries): Boundary to generate data on.
        initials (initials): Initial conditions to generate data on.
        n_clp (int): Number of collocation points.
        n_bc (int): Number of boundary condition points.
        n_ic (int): Number of initial condition points.
        n_sensors (int): Number of sensors to sample u with.
      """

      super().__init__(domain, boundaries, n_clp, n_bc, n_sensors)
      self.set_data_type(4)
      self._initials = initials
      self._n_iv = n_ic
      self._icp = initials.sampleInitials(n_ic)

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