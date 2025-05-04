from .domain import domain
import numpy as np
from pyDOE import lhs

class timeXinterval(domain):

    def __init__(self, tx_bdry):
        self._t_bdry = tx_bdry[0]
        self._x_bdry = tx_bdry[1]

        return