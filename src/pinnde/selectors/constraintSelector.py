import tensorflow as tf
import warnings
from ..domain.Time_NRect import Time_NRect
from ..domain.NRect import NRect

def constraintSelector(domain, boundaries, eqns, initials=None):

    if len(eqns) > 1:
        warnings.warn("Only single equations can be hard constrained currently")
        return

    if boundaries.get_bdry_type() == 3:
        warnings.warn("Neumann boundaries cannot be hard constrained currently")
        return

    if isinstance(domain, Time_NRect):
        inits = initials.get_lambdas()
        if domain.get_dim() != 1:
            warnings.warn("Only 1+1 Equations can currently be hard constrained from 1+N")
            return

        # Periodic
        t = domain.get_timeRange()
        if boundaries.get_bdry_type() == 1:        
            if len(inits) == 1:
                return lambda x: inits[0](x[1]) + (x[0]-t[0])/(t[1]-t[0])*x[2]
            elif len(inits) == 2:
                return lambda x: inits[0](x[1]) + inits[1](x[1])*(x[0] - t[0]) \
                               +  (((x[0]-t[0])/(t[1]-t[0]))**2)*x[2]
            elif len(inits) == 3:
                return lambda x: inits[0](x[1]) + inits[1](x[1])*(x[0] - t[0]) \
                               + inits[2](x[1])*((x[0] - t[0])**2) \
                               +  (((x[0]-t[0])/(t[1]-t[0]))**3)*x[2]
            else:
                warnings.warn("Only up to order 3 of t equations have been hard constrained with periodic boundaries")
                return
            
        # Dricihlet
        elif boundaries.get_bdry_type()  == 2:
            if len(inits) == 1:
                xleft_bound, xright_bound = boundaries.get_lambdas()[0], boundaries.get_lambdas()[1]
                t0, t1 = t[0], t[1]
                x0, x1 = domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0]
                return lambda x: (1-((x[0]-t0)/(t1-t0)))*inits[0](x[1]) + \
                                ((x[0]-t0)/(t1-t0))*x[2] + (1-((x[1]-x0)/(x1-x0)))* \
                                (xleft_bound(x[0]) - ((1-((x[0]-t0)/(t1-t0)))*xleft_bound(t0) + \
                                ((x[0]-t0)/(t1-t0)*xleft_bound(t1)))) + \
                                ((x[1]-x0)/(x1-x0)) * \
                                (xright_bound(x[0]) - ((1-((x[0]-t0)/(t1-t0)))*xright_bound(t0) + \
                                ((x[0]-t0)/(t1-t0)*xright_bound(t1)))) \
                                + ((x[0]-t0)/(t1-t0))*(1-((x[0]-t0)/(t1-t0)))* \
                                ((x[1]-x0)/(x1-x0))*(1-((x[1]-x0)/(x1-x0)))*x[2]

            else:
                warnings.warn("Only up to order 1 of t equations have been hard constrained with Dirichlet boundaries")
                return
            
    elif isinstance(domain, NRect):

        if boundaries.get_bdry_type()  == 2:
            if domain.get_dim() == 2:
                x0, y0 = domain.get_min_dim_vals()
                x1, y1 = domain.get_max_dim_vals()
                xleft_bound, xright_bound = boundaries.get_lambdas()[0], boundaries.get_lambdas()[1]
                ylower_bound, yupper_bound = boundaries.get_lambdas()[2], boundaries.get_lambdas()[3]
                
                return lambda x: (1-((x[0]-x0)/(x1-x0)))*xleft_bound(x0, x[1]) + \
                              ((x[0]-x0)/(x1-x0))*xright_bound(x1, x[1]) + (1-((x[1]-y0)/(y1-y0)))* \
                              (ylower_bound(x[0], y0) - ((1-((x[0]-x0)/(x1-x0)))*ylower_bound(x0, y0) + \
                               ((x[0]-x0)/(x1-x0)*ylower_bound(x1, y0)))) + \
                               ((x[1]-y0)/(y1-y0)) * \
                               (yupper_bound(x[0], y1) - ((1-((x[0]-x0)/(x1-x0)))*yupper_bound(x0, y1) + \
                               ((x[0]-x0)/(x1-x0)*yupper_bound(x1, y1)))) + \
                               ((x[0]-x0)/(x1-x0))*(1-((x[0]-x0)/(x1-x0)))* \
                               ((x[1]-y0)/(y1-y0))*(1-((x[1]-y0)/(y1-y0)))*x[2]
    
        else: 
            raise ValueError("Hard constrained currently not supported for this")

    return
    