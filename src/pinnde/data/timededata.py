from .dedata import dedata

class timededata(dedata):

  def __init__(self, domain, boundaries, initials,
                 n_clp=1000, n_bc=100, n_iv=100):

      super().__init__(domain, boundaries, n_clp, n_bc)
      self._initials = initials
      self._n_iv = n_iv
      self._icp = initials.sampleInitials(n_iv)

  def get_icp(self):
    return self._icp

  def get_initials(self):
    return self._initials