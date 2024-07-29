import numpy as np
import tensorflow as tf
from pyDOE import lhs
import inspect

def setup_boundries_periodic_tx():
    """
    Boundary function for setting up periodic boundaries for a PDE with independent variables x and y. 
    
    No Arguments. All returns information returned in single list

    Returns:
        flag (string): string to identify what boundary condition was used internally.    
    """
    return ["periodic_timeDependent"]


"""return flag for model ro understand what to build"""
def setup_boundries_periodic_xy():
    """
    Boundary function for setting up periodic boundaries for a PDE with independent variables t and x. 
    
    No Arguments. All returns information returned in single list

    Returns:
        flag (string): string to identify what boundary condition was used internally.
    """
    return ["periodic_timeIndependent"]


""""User responsable for boundries to line up correctly"""
def setup_boundries_dirichlet_tx(t_bdry, x_bdry, N_bc=100, all_boundries_cond=None, xleft_boundry_cond=None, 
                                            xright_boundry_cond=None):
    """
    Boundary function for setting up dirichlet boundaries for a PDE with independent variables t and x. 
    
    Args:
        t_bdry (list): list of two elements, the interval of t to be solved on.
        x_bdry (list): list of two elements, the interval of x to be solved on.
        N_bc (int): Number of randomly sampled collocation points along each boundary to use in training
        all_boundries_cond (lambda): function specifying condition on both xleft and xright boundary.
            If declared leave xleft_boundry_cond and xright_boundry_cond undeclared. lambda **must** be of 1 variable.
        xleft_boundry_cond (lambda): function specifying condition on xleft boundary.
            If declared then declare xright_boundry_cond and leave all_boundries_cond undeclared. lambda **must** be of 1 variable.
        xright_boundry_cond (lambda): function specifying condition on xright boundary.
            If declared then declare xright_boundry_cond and leave all_boundries_cond undeclared. lambda **must** be of 1 variable.

    All returns information returned in single list.

    Returns:
        flag (string): string to identify what boundary condition was used internally. 
        bcs (np_column_stack): stack of three lists; boundary points along t, along x, and functional values u along each boundary.
        xleft_boundry_cond (lambda): input lambda or all_boundries_cond, returned for when hard constrainting
        xright_boundry_cond (lambda): input lambda or all_boundries_cond, returned for when hard constrainting
    """
    tx_min = np.array([t_bdry[0], x_bdry[0]])
    tx_max = np.array([t_bdry[1], x_bdry[1]])
    xleft, xright = x_bdry[0], x_bdry[1]

    if all_boundries_cond != None:
        if len((inspect.getfullargspec(all_boundries_cond))[0]) != 1:
            raise ValueError("Lambda function for boundary condition must be of 1 variable (t)")

        t_bc_points, x_bc_points = [], []

        # Left boundary
        tbc = tx_min[:1] + (tx_max[:1] - tx_min[:1])*lhs(1, N_bc)
        xbc = xleft + 0*tbc

        x_bc_points.append(xbc), t_bc_points.append(tbc)

        # Right boundary
        tbc = tx_min[:1] + (tx_max[:1] - tx_min[:1])*lhs(1, N_bc)
        xbc = xright + 0*tbc

        x_bc_points.append(xbc), t_bc_points.append(tbc)

        x_bc_points = np.array(x_bc_points).flatten()
        t_bc_points = np.array(t_bc_points).flatten()
        u_bc_points = all_boundries_cond(t_bc_points)

        # All boundary data
        bcs = np.column_stack([t_bc_points, x_bc_points,
                                u_bc_points]).astype(np.float64)
       
        xleft_boundry_cond = all_boundries_cond
        xright_boundry_cond = all_boundries_cond
        
    elif all_boundries_cond == None:
        if len((inspect.getfullargspec(xleft_boundry_cond))[0]) != 1:
            raise ValueError("Lambda function for xleft boundary condition must be of 1 variable (t)")
        if len((inspect.getfullargspec(xright_boundry_cond))[0]) != 1:
            raise ValueError("Lambda function for xrigth boundary condition must be of 1 variable (t)")
        
        x_bc_points, t_bc_points = [], []
        u_bc_points = []

        # Left boundary
        tbc = tx_min[:1] + (tx_max[:1] - tx_min[:1])*lhs(1, N_bc)
        xbc = xleft + 0*tbc
        ubc = xleft_boundry_cond(tbc)

        x_bc_points.append(xbc), t_bc_points.append(tbc), u_bc_points.append(ubc)

        # Right boundary
        tbc = tx_min[:1] + (tx_max[:1] - tx_min[:1])*lhs(1, N_bc)
        xbc = xright + 0*tbc
        ubc = xright_boundry_cond(tbc)

        x_bc_points.append(xbc), t_bc_points.append(tbc), u_bc_points.append(ubc)


        x_bc_points = np.array(x_bc_points).flatten()
        t_bc_points = np.array(t_bc_points).flatten()
        u_bc_points = np.array(u_bc_points).flatten()
        u_bc_points = tf.reshape(u_bc_points, (N_bc*2,))

        # All boundary data
        bcs = np.column_stack([t_bc_points, x_bc_points,
                                u_bc_points]).astype(np.float64)
        
    return ["dirichlet_timeDependent", bcs, N_bc, xleft_boundry_cond, xright_boundry_cond]


def setup_boundries_dirichlet_xy(x_bdry, y_bdry, N_bc=100, all_boundries_cond=None, xleft_boundry_cond=None, 
                                              xright_boundry_cond=None, ylower_boundry_cond=None, yupper_boundry_cond=None):
    """
    Boundary function for setting up dirichlet boundaries for a PDE with independent variables t and x. 
    
    Args:
        x_bdry (list): list of two elements, the interval of x to be solved on.
        y_bdry (list): list of two elements, the interval of y to be solved on.
        N_bc (int): Number of randomly sampled collocation points along each boundary to use in training
        all_boundries_cond (lambda): function specifying condition on both xleft and xright boundary.
            If declared leave xleft_boundry_cond, xright_boundry_cond, ylower_boundry_cond, yupper_boundry_cond undeclared.
            lambda **must** be of 2 variables (x,y) even if one not used along boundary.
        xleft_boundry_cond (lambda): function specifying condition on xleft boundary.
            If declared then declare xright_boundry_cond, ylower_boundry_cond, yupper_boundry_cond,
            and leave all_boundries_cond undeclared. lambda **must** be of 2 variables (x,y) even if one not used along specific boundary.
        xright_boundry_cond (lambda): function specifying condition on xright boundary.
            If declared then declare xleft_boundry_cond, ylower_boundry_cond, yupper_boundry_cond,
            and leave all_boundries_cond undeclared. lambda **must** be of 2 variables (x,y) even if one not used along specific boundary.
        ylower_boundry_cond (lambda): function specifying condition on ylower boundary.
            If declared then declare xleft_boundry_cond, xright_boundry_cond, yupper_boundry_cond,
            and leave all_boundries_cond undeclared. lambda **must** be of 2 variables (x,y) even if one not used along specific boundary.
        yupper_boundry_cond (lambda): function specifying condition on yupper boundary.
            If declared then declare xleft_boundry_cond, xright_boundry_cond, ylower_boundry_cond,
            and leave all_boundries_cond undeclared. lambda **must** be of 2 variables (x,y) even if one not used along specific boundary.

    All returns information returned in single list.

    Returns:
        flag (string): string to identify what boundary condition was used internally. 
        bcs (np_column_stack): stack of three lists; boundary points along x, along y, and functional values u along each boundary.
        xleft_boundry_cond (lambda): input lambda or all_boundries_cond, returned for when hard constrainting
        xright_boundry_cond (lambda): input lambda or all_boundries_cond, returned for when hard constrainting
        ylower_boundry_cond (lambda): input lambda or all_boundries_cond, returned for when hard constrainting
        yupper_boundry_cond (lambda): input lambda or all_boundries_cond, returned for when hard constrainting
    """

    xy_min = np.array([x_bdry[0], y_bdry[0]])
    xy_max = np.array([x_bdry[1], y_bdry[1]])
    xleft, xright = x_bdry[0], x_bdry[1]
    ylower, yupper = y_bdry[0], y_bdry[1]

    if all_boundries_cond != None:
        if len((inspect.getfullargspec(all_boundries_cond))[0]) != 2:
            raise ValueError("Lambda function for boundary condition must be of 2 variables (x,y) even if one variable not used \
                             i.e. lambda x, y: x**2")
        
        x_bc_points, y_bc_points = [], []

        # Left boundary
        ybc = xy_min[:1] + (xy_max[:1] - xy_min[:1])*lhs(1, N_bc)
        xbc = xleft + 0*ybc

        x_bc_points.append(xbc), y_bc_points.append(ybc)

        # Right boundary
        ybc = xy_min[:1] + (xy_max[:1] - xy_min[:1])*lhs(1, N_bc)
        xbc = xright + 0*ybc

        x_bc_points.append(xbc), y_bc_points.append(ybc)

        # Lower boundary
        xbc = xy_min[1:] + (xy_max[1:] - xy_min[1:])*lhs(1, N_bc)
        ybc = ylower + 0*xbc

        x_bc_points.append(xbc), y_bc_points.append(ybc)

        # Upper boundary
        xbc = xy_min[1:] + (xy_max[1:] - xy_min[1:])*lhs(1, N_bc)
        ybc = yupper + 0*xbc

        x_bc_points.append(xbc), y_bc_points.append(ybc)

        x_bc_points = np.array(x_bc_points).flatten()
        y_bc_points = np.array(y_bc_points).flatten()
        u_bc_points = all_boundries_cond(x_bc_points, y_bc_points)

        # All boundary data
        bcs = np.column_stack([x_bc_points, y_bc_points,
                                u_bc_points]).astype(np.float64)
        
        xleft_boundry_cond = all_boundries_cond
        xright_boundry_cond = all_boundries_cond
        ylower_boundry_cond = all_boundries_cond
        yupper_boundry_cond = all_boundries_cond
        
    elif all_boundries_cond == None:
        if len((inspect.getfullargspec(xleft_boundry_cond))[0]) != 2:
            raise ValueError("Lambda function for xleft boundary condition must be of 2 variables (x,y) even if one variable not used,\
                              i.e lambda x, y: y**2")
        if len((inspect.getfullargspec(xright_boundry_cond))[0]) != 2:
            raise ValueError("Lambda function for xright boundary condition must be of 2 variables (x,y) even if one variable not used,\
                              i.e lambda x, y: y**2")
        if len((inspect.getfullargspec(ylower_boundry_cond))[0]) != 2:
            raise ValueError("Lambda function for ylower boundary condition must be of 2 variables (x,y) even if one variable not used,\
                              i.e lambda x, y: x**2")
        if len((inspect.getfullargspec(yupper_boundry_cond))[0]) != 2:
            raise ValueError("Lambda function for yupper boundary condition must be of 2 variables (x,y) even if one variable not used,\
                              i.e lambda x, y: x**2")
        
        x_bc_points, y_bc_points = [], []
        u_bc_points = []

        # Left boundary
        ybc = xy_min[:1] + (xy_max[:1] - xy_min[:1])*lhs(1, N_bc)
        xbc = xleft + 0*ybc
        ubc = xleft_boundry_cond(xbc, ybc)

        x_bc_points.append(xbc), y_bc_points.append(ybc), u_bc_points.append(ubc)

        # Right boundary
        ybc = xy_min[:1] + (xy_max[:1] - xy_min[:1])*lhs(1, N_bc)
        xbc = xright + 0*ybc
        ubc = xright_boundry_cond(xbc, ybc)

        x_bc_points.append(xbc), y_bc_points.append(ybc), u_bc_points.append(ubc)

        # Lower boundary
        xbc = xy_min[1:] + (xy_max[1:] - xy_min[1:])*lhs(1, N_bc)
        ybc = ylower + 0*xbc
        ubc = ylower_boundry_cond(xbc, ybc)

        x_bc_points.append(xbc), y_bc_points.append(ybc), u_bc_points.append(ubc)

        # Upper boundary
        xbc = xy_min[1:] + (xy_max[1:] - xy_min[1:])*lhs(1, N_bc)
        ybc = yupper + 0*xbc
        ubc = yupper_boundry_cond(xbc, ybc)

        x_bc_points.append(xbc), y_bc_points.append(ybc), u_bc_points.append(ubc)

        x_bc_points = np.array(x_bc_points).flatten()
        y_bc_points = np.array(y_bc_points).flatten()
        u_bc_points = np.array(u_bc_points).flatten()
        u_bc_points = tf.reshape(u_bc_points, (N_bc*4,))
    

        # All boundary data
        bcs = np.column_stack([x_bc_points, y_bc_points,
                                u_bc_points]).astype(np.float64)
        
    return ["dirichlet_timeIndependent", bcs, N_bc, xleft_boundry_cond, xright_boundry_cond, ylower_boundry_cond, yupper_boundry_cond]

def setup_boundries_neumann_tx(t_bdry, x_bdry, N_bc=100, all_boundries_cond=None, xleft_boundry_cond=None, 
                                            xright_boundry_cond=None):
    """
    Boundary function for setting up dirichlet boundaries for a PDE with independent variables t and x. 
    
    Args:
        t_bdry (list): list of two elements, the interval of t to be solved on.
        x_bdry (list): list of two elements, the interval of x to be solved on.
        N_bc (int): Number of randomly sampled collocation points along each boundary to use in training
        all_boundries_cond (lambda): function specifying condition on both xleft and xright boundary.
            If declared leave xleft_boundry_cond and xright_boundry_cond undeclared. lambda **must** be of 1 variable.
        xleft_boundry_cond (lambda): function specifying condition on xleft boundary.
            If declared then declare xright_boundry_cond and leave all_boundries_cond undeclared. lambda **must** be of 1 variable.
        xright_boundry_cond (lambda): function specifying condition on xright boundary.
            If declared then declare xright_boundry_cond and leave all_boundries_cond undeclared. lambda **must** be of 1 variable.

    All returns information returned in single list.

    Returns:
        flag (string): string to identify what boundary condition was used internally. 
        bcs (np_column_stack): stack of three lists; boundary points along t, along x, and functional values u along each boundary.
        xleft_boundry_cond (lambda): input lambda or all_boundries_cond, returned for when hard constrainting
        xright_boundry_cond (lambda): input lambda or all_boundries_cond, returned for when hard constrainting
    """
    tx_min = np.array([t_bdry[0], x_bdry[0]])
    tx_max = np.array([t_bdry[1], x_bdry[1]])
    xleft, xright = x_bdry[0], x_bdry[1]

    if all_boundries_cond != None:
        if len((inspect.getfullargspec(all_boundries_cond))[0]) != 1:
            raise ValueError("Lambda function for boundary condition must be of 1 variable (t)")
        
        x_bound_points, t_bc_points = [], []

        # Left boundary
        tbc = tx_min[:1] + (tx_max[:1] - tx_min[:1])*lhs(1, N_bc)
        xbc = xleft + 0*tbc

        x_bound_points.append(xbc), t_bc_points.append(tbc)

        # Right boundary
        tbc = tx_min[:1] + (tx_max[:1] - tx_min[:1])*lhs(1, N_bc)
        xbc = xright + 0*tbc

        x_bound_points.append(xbc), t_bc_points.append(tbc)

        x_bound_points = np.array(x_bound_points).flatten()
        t_bc_points = np.array(t_bc_points).flatten()
        ux_bc_points = all_boundries_cond(t_bc_points)

        # All boundary data
        bcs = np.column_stack([t_bc_points, x_bound_points,
                                ux_bc_points]).astype(np.float64)
       
        xleft_boundry_cond = all_boundries_cond
        xright_boundry_cond = all_boundries_cond
        
    elif all_boundries_cond == None:
        if len((inspect.getfullargspec(xleft_boundry_cond))[0]) != 1:
            raise ValueError("Lambda function for xleft boundary condition must be of 1 variable (t)")
        if len((inspect.getfullargspec(xright_boundry_cond))[0]) != 1:
            raise ValueError("Lambda function for xright boundary condition must be of 1 variable (t)")
        
        x_bound_points, t_bc_points = [], []
        ux_bc_points = []

        # Left boundary
        tbc = tx_min[:1] + (tx_max[:1] - tx_min[:1])*lhs(1, N_bc)
        xbc = xleft + 0*tbc
        ubc = xleft_boundry_cond(tbc)

        x_bound_points.append(xbc), t_bc_points.append(tbc), ux_bc_points.append(ubc)

        # Right boundary
        tbc = tx_min[:1] + (tx_max[:1] - tx_min[:1])*lhs(1, N_bc)
        xbc = xright + 0*tbc
        ubc = xright_boundry_cond(tbc)

        x_bound_points.append(xbc), t_bc_points.append(tbc), ux_bc_points.append(ubc)


        x_bound_points = np.array(x_bound_points).flatten()
        t_bc_points = np.array(t_bc_points).flatten()
        ux_bc_points = np.array(ux_bc_points).flatten()
        ux_bc_points = tf.reshape(ux_bc_points, (N_bc*2,))

        # All boundary data
        bcs = np.column_stack([t_bc_points, x_bound_points,
                                ux_bc_points]).astype(np.float64)
    
    
    return ["neumann_timeDependent", bcs, N_bc, xleft_boundry_cond, xright_boundry_cond]

""""User responsable for boundries to line up correctly"""
def setup_boundries_neumann_xy(x_bdry, y_bdry, N_bc=100, all_boundries_cond=None, xleft_boundry_cond=None, 
                                              xright_boundry_cond=None, ylower_boundry_cond=None, yupper_boundry_cond=None):
    """
    Boundary function for setting up dirichlet boundaries for a PDE with independent variables t and x. 
    
    Args:
        x_bdry (list): list of two elements, the interval of x to be solved on.
        y_bdry (list): list of two elements, the interval of y to be solved on.
        N_bc (int): Number of randomly sampled collocation points along each boundary to use in training
        all_boundries_cond (lambda): function specifying condition on both xleft and xright boundary.
            If declared leave xleft_boundry_cond, xright_boundry_cond, ylower_boundry_cond, yupper_boundry_cond undeclared.
            lambda **must** be of 2 variables (x,y) even if one not used along boundary.
        xleft_boundry_cond (lambda): function specifying condition on xleft boundary.
            If declared then declare xright_boundry_cond, ylower_boundry_cond, yupper_boundry_cond,
            and leave all_boundries_cond undeclared. lambda **must** be of 2 variables (x,y) even if one not used along specific boundary.
        xright_boundry_cond (lambda): function specifying condition on xright boundary.
            If declared then declare xleft_boundry_cond, ylower_boundry_cond, yupper_boundry_cond,
            and leave all_boundries_cond undeclared. lambda **must** be of 2 variables (x,y) even if one not used along specific boundary.
        ylower_boundry_cond (lambda): function specifying condition on ylower boundary.
            If declared then declare xleft_boundry_cond, xright_boundry_cond, yupper_boundry_cond,
            and leave all_boundries_cond undeclared. lambda **must** be of 2 variables (x,y) even if one not used along specific boundary.
        yupper_boundry_cond (lambda): function specifying condition on yupper boundary.
            If declared then declare xleft_boundry_cond, xright_boundry_cond, ylower_boundry_cond,
            and leave all_boundries_cond undeclared. lambda **must** be of 2 variables (x,y) even if one not used along specific boundary.

    All returns information returned in single list.

    Returns:
        flag (string): string to identify what boundary condition was used internally. 
        bcs (np_column_stack): stack of 6; sampled boundary points along x and y, list of just repeated boundary values for
            x and y, and functional values u along both x boundries and y boundries. Seperated for derivatives in training for 
            neumann constrainting.
        xleft_boundry_cond (lambda): input lambda or all_boundries_cond, returned for when hard constrainting
        xright_boundry_cond (lambda): input lambda or all_boundries_cond, returned for when hard constrainting
        ylower_boundry_cond (lambda): input lambda or all_boundries_cond, returned for when hard constrainting
        yupper_boundry_cond (lambda): input lambda or all_boundries_cond, returned for when hard constrainting
    """

    xy_min = np.array([x_bdry[0], y_bdry[0]])
    xy_max = np.array([x_bdry[1], y_bdry[1]])
    xleft, xright = x_bdry[0], x_bdry[1]
    ylower, yupper = y_bdry[0], y_bdry[1]

    if all_boundries_cond != None:
        if len((inspect.getfullargspec(all_boundries_cond))[0]) != 2:
            raise ValueError("Lambda function for boundary condition must be of 2 variables (x,y) even if one variable not used \
                             i.e. lambda x, y: x**2")
        
        x_bc_points, y_bc_points = [], []
        x_bound_points, y_bound_points = [], []
        
        # Left boundary
        ybc = xy_min[:1] + (xy_max[:1] - xy_min[:1])*lhs(1, N_bc)
        xbc = xleft + 0*ybc

        x_bound_points.append(xbc), y_bc_points.append(ybc)

        # Right boundary
        ybc = xy_min[:1] + (xy_max[:1] - xy_min[:1])*lhs(1, N_bc)
        xbc = xright + 0*ybc

        x_bound_points.append(xbc), y_bc_points.append(ybc)

        # Lower boundary
        xbc = xy_min[1:] + (xy_max[1:] - xy_min[1:])*lhs(1, N_bc)
        ybc = ylower + 0*xbc

        x_bc_points.append(xbc), y_bound_points.append(ybc)

        # Upper boundary
        xbc = xy_min[1:] + (xy_max[1:] - xy_min[1:])*lhs(1, N_bc)
        ybc = yupper + 0*xbc

        x_bc_points.append(xbc), y_bound_points.append(ybc)

        x_bc_points = np.array(x_bc_points).flatten()
        y_bc_points = np.array(y_bc_points).flatten()
        x_bound_points = np.array(x_bound_points).flatten()
        y_bound_points = np.array(y_bound_points).flatten()
        ux_bc_points = all_boundries_cond(x_bound_points, y_bc_points)
        uy_bc_points = all_boundries_cond(x_bc_points, y_bound_points)

        # All boundary data
        bcs = np.column_stack([x_bound_points, y_bound_points, x_bc_points, y_bc_points,
                                ux_bc_points, uy_bc_points]).astype(np.float64)
        
        xleft_boundry_cond = all_boundries_cond
        xright_boundry_cond = all_boundries_cond
        ylower_boundry_cond = all_boundries_cond
        yupper_boundry_cond = all_boundries_cond
        
    elif all_boundries_cond == None:
        if len((inspect.getfullargspec(xleft_boundry_cond))[0]) != 2:
            raise ValueError("Lambda function for xleft boundary condition must be of 2 variables (x,y) even if one variable not used,\
                              i.e lambda x, y: y**2")
        if len((inspect.getfullargspec(xright_boundry_cond))[0]) != 2:
            raise ValueError("Lambda function for xright boundary condition must be of 2 variables (x,y) even if one variable not used,\
                              i.e lambda x, y: y**2")
        if len((inspect.getfullargspec(ylower_boundry_cond))[0]) != 2:
            raise ValueError("Lambda function for ylower boundary condition must be of 2 variables (x,y) even if one variable not used,\
                              i.e lambda x, y: x**2")
        if len((inspect.getfullargspec(yupper_boundry_cond))[0]) != 2:
            raise ValueError("Lambda function for yupper boundary condition must be of 2 variables (x,y) even if one variable not used,\
                              i.e lambda x, y: x**2")
        
        x_bc_points, y_bc_points = [], []
        x_bound_points, y_bound_points = [], []
        ux_bc_points, uy_bc_points = [], []

        # Left boundary
        ybc = xy_min[:1] + (xy_max[:1] - xy_min[:1])*lhs(1, N_bc)
        xbc = xleft + 0*ybc
        ubc = xleft_boundry_cond(xbc, ybc)

        x_bound_points.append(xbc), y_bc_points.append(ybc), ux_bc_points.append(ubc)

        # Right boundary
        ybc = xy_min[:1] + (xy_max[:1] - xy_min[:1])*lhs(1, N_bc)
        xbc = xright + 0*ybc
        ubc = xright_boundry_cond(xbc, ybc)

        x_bound_points.append(xbc), y_bc_points.append(ybc), ux_bc_points.append(ubc)

        # Lower boundary
        xbc = xy_min[1:] + (xy_max[1:] - xy_min[1:])*lhs(1, N_bc)
        ybc = ylower + 0*xbc
        ubc = ylower_boundry_cond(xbc, ybc)

        x_bc_points.append(xbc), y_bound_points.append(ybc), uy_bc_points.append(ubc)

        # Upper boundary
        xbc = xy_min[1:] + (xy_max[1:] - xy_min[1:])*lhs(1, N_bc)
        ybc = yupper + 0*xbc
        ubc = yupper_boundry_cond(xbc, ybc)

        x_bc_points.append(xbc), y_bound_points.append(ybc), uy_bc_points.append(ubc)

        x_bc_points = np.array(x_bc_points).flatten()
        y_bc_points = np.array(y_bc_points).flatten()
        x_bound_points = np.array(x_bound_points).flatten()
        y_bound_points = np.array(y_bound_points).flatten()
        ux_bc_points = np.array(ux_bc_points).flatten()
        ux_bc_points = tf.reshape(ux_bc_points, (N_bc*2,))
        uy_bc_points = np.array(uy_bc_points).flatten()
        uy_bc_points = tf.reshape(uy_bc_points, (N_bc*2,))
    

        # All boundary data
        bcs = np.column_stack([x_bound_points, y_bound_points, x_bc_points, y_bc_points,
                                ux_bc_points, uy_bc_points]).astype(np.float64)
    
    return ["neumann_timeIndependent", bcs, N_bc, xleft_boundry_cond, xright_boundry_cond, ylower_boundry_cond, yupper_boundry_cond]

