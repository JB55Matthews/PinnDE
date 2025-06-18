from .dondata import dondata

class timedondata(dondata):

  def __init__(self, domain, boundaries, initials,
                 n_clp=10000, n_bc=600, n_ic=600, n_sensors=1000):

      super().__init__(domain, boundaries, n_clp, n_bc, n_sensors)
      self.set_data_type(4)
      self._initials = initials
      self._n_iv = n_ic
      self._icp = initials.sampleInitials(n_ic)

  def get_icp(self):
    return self._icp

  def get_initials(self):
    return self._initials
  
  def get_n_ic(self):
    return self._n_iv