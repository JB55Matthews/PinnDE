from abc import ABC, abstractmethod
# from .dirichlet import dirichlet
# from .neumann import neumann
# from .periodic import periodic

class boundaries(ABC):
  # periodic = 1
  # dirichlet = 2
  # neumann = 3

  def __init__(self, domain, type):
    self._domain = domain
    self._bdry_type = type

  def get_domain(self):
    return self._domain

  def get_bdry_type(self):
    return self._bdry_type

  def set_bdry_type(self, bdry):
     # periodic = 1
    # dirichlet = 2
    # neumann = 3
    self._bdry_type = bdry

  @abstractmethod
  def boundaryPoints(self, n_bc):
    # must implement taking domain.sampleBoundary and adding boundary function points if any
    return