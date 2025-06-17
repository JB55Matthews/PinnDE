from .data import data

class pinndata(data):

    def __init__(self, domain, boundaries, n_clp=1000, n_bc=100):
        super().__init__(1)
        self._domain = domain
        self._boundaries = boundaries
        self._n_clp = n_clp
        self._n_bc = n_bc
        self._clp = domain.sampleDomain(n_clp)
        self._bcp = boundaries.boundaryPoints(n_bc)
        return

    def get_clp(self):
      return self._clp

    def get_bcp(self):
      return self._bcp

    def get_domain(self):
        return self._domain

    def get_boundaries(self):
        return self._boundaries
    
    def get_n_clp(self):
      return self._n_clp

    def get_n_bc(self):
      return self._n_bc