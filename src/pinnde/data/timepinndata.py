from .pinndata import pinndata

class timepinndata(pinndata):

  def __init__(self, domain, boundaries, initials,
                 n_clp=1000, n_bc=100, n_ic=100):

      super().__init__(domain, boundaries, n_clp, n_bc)
      self.set_data_type(2)
      self._initials = initials
      self._n_iv = n_ic
      self._icp = initials.sampleInitials(n_ic)

  def get_icp(self):
    return self._icp

  def get_initials(self):
    return self._initials
  
  def get_n_ic(self):
    return self._n_iv