import tensorflow as tf
import numpy as np
from .adaptives import adaptives
from ..selectors.adaptSampleSelector import adaptSampleSelector

class RAD(adaptives):
  """
  Class which implements stategy to call Residule-based Adaptive Refinement, based on 
    [A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks](https://arxiv.org/abs/2207.10289)
  """

  def __init__(self, frequency, k=2, c=1, samplefactor=3):
    """
    Args:
      frequency (int): How many epochs between sampling new collocation points.
      k (int): Hyperparameter which affects sampling. Please see paper [here](https://arxiv.org/abs/2207.10289) 
      c (int): Hyperparameter which affects sampling. Please see cited paper [here](https://arxiv.org/abs/2207.10289) 
      samplefactor (int): Factor of number of collocation points to sample to choose new
        distribution from.
    """
    self._frequency = frequency
    self._k = k
    self._c = c
    self._samplefactor = samplefactor

  def get_frequency(self):
    """
    Returns:
      (int): How many epochs between sampling new collocation points.
    """
    return self._frequency

  def get_k(self):
    """
    Returns:
      (int): K value used
    """
    return self._k

  def get_c(self):
    """
    Returns:
      (int): C value used
    """
    return self._c

  def get_samplefactor(self):
    """
    Returns:  
      (int): Factor of number of collocation points to sample to choose new
        distribution from.
    """
    return self._samplefactor

  def AdaptiveStrategy(self, model, domain, data,  clps, ds_data, ds, i):
    """
        Sampling stategy to call Residule-based Adaptive Distribtuion.

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
      points = domain.sampleDomain(self._samplefactor*N_clp)
      clps_group = []
      for i in range(points.shape[1]):
          clps_group.append(points[:,i:i+1])
      f = adaptSampleSelector(model, clps_group, data)
      err_eq = np.absolute(f)

      err_eq = np.power(err_eq, self._k)
      err = np.mean(err_eq)
      p = ((err_eq/err)+self._c)
      A = np.sum(p)
      phat = p/A
      cdf = np.cumsum(phat)

      sample = np.random.rand()
      points_id = np.argmax(cdf>sample)

      new_point_inner = []
      for k in range(points.shape[1]):
          new_point_inner.append(points[points_id][k])
      new_clps = [new_point_inner]
      points = np.delete(points, [points_id], 0)
      cdf = np.delete(cdf, [points_id], 0)

      for j in range(N_clp-1):
        sample = np.random.rand()
        points_id = np.argmax(cdf>sample)
        new_point_inner = []
        for k in range(points.shape[1]):
          new_point_inner.append(points[points_id][k])
        new_point = [new_point_inner]
        new_clps = tf.concat([new_clps, new_point], 0)
        points = np.delete(points, [points_id], 0)
        cdf = np.delete(cdf, [points_id], 0)

      ds_clp = tf.data.Dataset.from_tensor_slices(new_clps)
      ds_clp = ds_clp.cache().shuffle(N_clp).batch(N_clp)

      ds_data.insert(0, ds_clp)

      ds = tf.data.Dataset.zip(tuple(ds_data))
      clps = new_clps

    return ds, clps