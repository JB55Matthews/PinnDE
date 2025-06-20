
class data():
    """
    Class for implementing data
    """

    def __init__(self, type):
        """
        Constructor for class

        Args:
            type (int): Internal type int of data.
        """
        self._data_type = type
        return
    
    def get_data_type(self):
        """
        Returns:
            (int): Data type.
        """
        return self._data_type
    
    def set_data_type(self, type):
        """
        Args:
            type (int): Data type.
        """
        self._data_type = type