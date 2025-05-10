
class data():

    def __init__(self, type):
        self._data_type = type
        return
    
    def get_data_type(self):
        return self._data_type
    
    def set_data_type(self, type):
        self._data_type = type