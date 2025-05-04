from .model import model

class pinn(model):

    def __init__(self, geometry, boundaries, initials, 
                 layers=40, units=4, inner_act="tahn",
                 out_act="linear", hard_constraint="false"):
        
        return
    
    def train(self, eqns, epochs, opt="adam", meta="false", adapt_pt="false"):
        return