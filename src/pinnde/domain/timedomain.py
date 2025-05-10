from .domain import domain
from abc import ABC, abstractmethod

class timedomain(domain):

    def __init__(self, dim, timeRange):
      super().__init__(dim)
      self._timeRange = timeRange

    def get_timeRange(self):
      return self._timeRange

    @abstractmethod
    def onInitial(self, point):
      pass