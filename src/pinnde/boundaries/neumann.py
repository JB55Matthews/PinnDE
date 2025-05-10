from .boundaries import boundaries

class neumann(boundaries):

    def __init__(self, domain, lambdas, n_bc):
        return