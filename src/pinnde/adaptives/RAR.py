import tensorflow as tf
import numpy as np
from .adaptives import adaptives
from ..selectors.adaptSampleSelector import adaptSampleSelector

class RAR(adaptives):
  """
  Class which implements stategy to call Residule-based Adaptive Refinement.
  """

  def __init__(self, frequency, pointsperfreq):
    """
    Args:
      frequency (int): How many epochs between sampling new collocation points.
      pointsperfreq (int): How many points to add in each sample.
    """
    self._frequency = frequency
    self._pointsperfreq = pointsperfreq

  def get_frequency(self):
    """
    Returns:
      (int): How many epochs between sampling new collocation points.
    """
    return self._frequency

  def get_pointsperfreq(self):
    """
    Returns:
      (int): How many points to add in each sample.
    """
    return self._pointsperfreq

  def AdaptiveStrategy(self, model, domain, data, clps, ds_data, ds, i):
    """
        Sampling stategy to call Residule-based Adaptive Refinement.

        Args:
            model (network): Tensorflow network.
            domain (domain): Domain class solving over.
            data (data): Data class solving with.
            clps (tensor): Current collocation points.
            ds_data (list): Data being packaged in training routine.
            ds (list): Current ds value of training routine.
            i (int): Iteration number.
            
    """
    if ((np.mod(i, self._frequency)==0) and (i != 0)):
      N_clp = data.get_n_clp()
      points = domain.sampleDomain(N_clp)
      clps_group = []
      for i in range(points.shape[1]):
          clps_group.append(points[:,i:i+1])
      f = adaptSampleSelector(model, clps_group, data)
      err_eq = np.absolute(f)

      for j in range(self._pointsperfreq):
        points_id = np.argmax(err_eq)
        new_point_inner = []
        for k in range(points.shape[1]):
          new_point_inner.append(points[points_id][k])
        new_point = [new_point_inner]
        clps = tf.concat([clps, new_point], 0)
        err_eq = np.delete(err_eq, points_id, 0)

      ds_clp = tf.data.Dataset.from_tensor_slices(clps)
      ds_clp = ds_clp.cache().shuffle(N_clp).batch(N_clp)

      ds_data.insert(0, ds_clp)

      ds = tf.data.Dataset.zip(tuple(ds_data))

    return ds, clps