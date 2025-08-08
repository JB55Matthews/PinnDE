from abc import ABC, abstractmethod

class adaptives(ABC):
    """
    Abstract class for adaptive point strategies
    """

    @abstractmethod
    def AdaptiveStrategy(self, model, domain, data, clps, ds_data, ds, i):
        """
        Sampling stategy to call

        Args:
            model (network): Tensorflow network.
            domain (domain): Domain class solving over.
            data (data): Data classs olving with.
            clps (tensor): Current collocation points.
            ds_data (list): Data being packaged in training routine.
            ds (list): Current ds value of training routine.
            i (int): Iteration number.
            
        """
        return