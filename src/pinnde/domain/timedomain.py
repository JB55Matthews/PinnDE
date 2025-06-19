from .domain import domain
from abc import ABC, abstractmethod

class timedomain(domain):
    """
    Class for domains with time components
    """

    def __init__(self, dim, timeRange):
      """
      Constructor for class

      Args:
        dim (int): Spatial dimension of domain.
        timeRange (list): Range of time to solve equation over, e.g, [0, 1].
      """
      super().__init__(dim)
      self._timeRange = timeRange

    def get_timeRange(self):
      """
      Returns: 
        (list): Range of time o solve equation over.
      """
      return self._timeRange

    @abstractmethod
    def onInitial(self, point):
      """
      Abstract method all timedomains must specify. Determines whether a point is an initial point.

      Args:
        point (list): Point in time+spatial dimensions of domain.

      Returns:
        (bool): True if point is an initial point, False otherwise.
      """
      pass