from .domain import domain
from pyDOE import lhs

class interval(domain):

    def __init__(self, t_bdry):
        self._t_bdry = t_bdry
        return
    
    def get_t_bdry(self):
        return self._t_bdry