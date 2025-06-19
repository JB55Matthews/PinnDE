from abc import ABC, abstractmethod
# from .dirichlet import dirichlet
# from .neumann import neumann
# from .periodic import periodic

class boundaries(ABC):
  """
  Abstract class for boundaries
  """
  # periodic = 1
  # dirichlet = 2
  # neumann = 3
  # odeicbc = 4

  def __init__(self, domain, type):
    """
    Constructor for class

    Args:
      domain (domain): Domain for differential equation, class implementing domain.
      type (int): Internal type given to each boundary. Self-created boundary should use 0.

    """
    self._domain = domain
    self._bdry_type = type

  def get_domain(self):
    """
    Returns:
      (domain): domain for boundary
    """
    return self._domain

  def get_bdry_type(self):
    """
    Returns:
      (int): Internal boundary type indicator.
    """
    return self._bdry_type

  def set_bdry_type(self, bdry):
     # periodic = 1
    # dirichlet = 2
    # neumann = 3
    """
    Args:
      bdry (int): Boundary type to set.
    """
    self._bdry_type = bdry

  @abstractmethod
  def boundaryPoints(self, n_bc):
    """
    Abstract function which must describe how to determine the boundary values of domain. Used in domain.sampleBoundary calls.

    Args:
      n_bc (int): Number of boundary condition points to sample.

    Returns sampled boundary values.
    """
    # must implement taking domain.sampleBoundary and adding boundary function points if any
    return