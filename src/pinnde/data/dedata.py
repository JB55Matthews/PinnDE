from .data import data

class dedata(data):

    def __init__(self, domain, boundaries, n_clp=1000, n_bc=100):
        self._domain = domain
        self._boundaries = boundaries
        self._n_clp = n_clp
        self._n_bc = n_bc
        return
    
    def get_domain(self):
        return self._domain
    
    def get_boundaries(self):
        return self._boundaries