import numpy as np
import tensorflow as tf
from pyDOE import lhs

"""returns flag for model to understand what to build"""
def setup_boundries_periodic_timeDependent():
    return ["periodic_timeDependent"]


"""return flag for model ro understand what to build"""
def setup_boundries_periodic_timeIndependent():
    return ["periodic_timeIndependent"]

def test_boundry(t_bdry, x_bdry, y_bdry,N_bc, all_boundries_cond):
    tx_min = np.array([t_bdry[0], x_bdry[0]])
    tx_max = np.array([t_bdry[1], x_bdry[1]])
    xleft, xright = x_bdry[0], x_bdry[1]

    if all_boundries_cond != None:
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
        
    return ["dirichlet_timeDependent", bcs, N_bc, xleft_boundry_cond, xright_boundry_cond]

""""User responsable for boundries to line up correctly"""
def setup_boundries_dirichlet_timeDependent(t_bdry, x_bdry, y_bdry, N_bc=100, all_boundries_cond=None, xleft_boundry_cond=None, 
                                            xright_boundry_cond=None, ylower_boundry_cond=None, yupper_boundry_cond=None):

    txy_min = np.array([t_bdry[0], x_bdry[0], y_bdry[0]])
    txy_max = np.array([t_bdry[1], x_bdry[1], y_bdry[1]])
    xleft, xright = x_bdry[0], x_bdry[1]
    ylower, yupper = y_bdry[0], y_bdry[1]

    if all_boundries_cond != None:
        t_bc_points, x_bc_points, y_bc_points = [], [], []

        # Left boundary
        tbc = txy_min[:1] + (txy_max[:1] - txy_min[:1])*lhs(1, N_bc)
        ybc = txy_min[2:] + (txy_max[2:] - txy_min[2:])*lhs(1, N_bc)
        xbc = xleft + 0*tbc

        x_bc_points.append(xbc), t_bc_points.append(tbc), y_bc_points.append(ybc)

        # Right boundary
        tbc = txy_min[:1] + (txy_max[:1] - txy_min[:1])*lhs(1, N_bc)
        ybc = txy_min[2:] + (txy_max[2:] - txy_min[2:])*lhs(1, N_bc)
        xbc = xright + 0*tbc

        x_bc_points.append(xbc), t_bc_points.append(tbc), y_bc_points.append(ybc)

        # Lower boundary
        tbc = txy_min[:1] + (txy_max[:1] - txy_min[:1])*lhs(1, N_bc)
        xbc = txy_min[1:2] + (txy_max[1:2] - txy_min[1:2])*lhs(1, N_bc)
        ybc = ylower + 0*xbc

        x_bc_points.append(xbc), y_bc_points.append(ybc), t_bc_points.append(tbc)

        # Upper boundary
        tbc = txy_min[:1] + (txy_max[:1] - txy_min[:1])*lhs(1, N_bc)
        xbc = txy_min[1:2] + (txy_max[1:2] - txy_min[1:2])*lhs(1, N_bc)
        ybc = yupper + 0*xbc

        x_bc_points.append(xbc), y_bc_points.append(ybc), t_bc_points.append(tbc)

        x_bc_points = np.array(x_bc_points).flatten()
        t_bc_points = np.array(t_bc_points).flatten()
        y_bc_points = np.array(y_bc_points).flatten()
        u_bc_points = all_boundries_cond(t_bc_points)

        # All boundary data
        bcs = np.column_stack([t_bc_points, x_bc_points, y_bc_points,
                                u_bc_points]).astype(np.float64)
       
        xleft_boundry_cond = all_boundries_cond
        xright_boundry_cond = all_boundries_cond
        ylower_boundry_cond = all_boundries_cond
        yupper_boundry_cond = all_boundries_cond
        
    elif all_boundries_cond == None:
        x_bc_points, t_bc_points, y_bc_points = [], [], []
        u_bc_points = []

        # Left boundary
        tbc = txy_min[:1] + (txy_max[:] - txy_min[:1])*lhs(1, N_bc)
        ybc = txy_min[2:] + (txy_max[2:] - txy_min[2:])*lhs(1, N_bc)
        xbc = xleft + 0*tbc
        ubc = xleft_boundry_cond(tbc, ybc)

        x_bc_points.append(xbc), t_bc_points.append(tbc), u_bc_points.append(ubc), y_bc_points.append(ybc)

        # Right boundary
        tbc = txy_min[:1] + (txy_max[:1] - txy_min[:1])*lhs(1, N_bc)
        ybc = txy_min[2:] + (txy_max[2:] - txy_min[2:])*lhs(1, N_bc)
        xbc = xright + 0*tbc
        ubc = xright_boundry_cond(tbc, ybc)

        x_bc_points.append(xbc), t_bc_points.append(tbc), u_bc_points.append(ubc), y_bc_points.append(ybc)

        # Lower boundary
        tbc = txy_min[:1] + (txy_max[:1] - txy_min[:1])*lhs(1, N_bc)
        xbc = txy_min[1:2] + (txy_max[1:2] - txy_min[1:2])*lhs(1, N_bc)
        ybc = ylower + 0*xbc
        ubc = ylower_boundry_cond(tbc, xbc)

        x_bc_points.append(xbc), y_bc_points.append(ybc), u_bc_points.append(ubc), t_bc_points.append(tbc)

        # Upper boundary
        tbc = txy_min[:1] + (txy_max[:1] - txy_min[:1])*lhs(1, N_bc)
        xbc = txy_min[1:2] + (txy_max[1:2] - txy_min[1:2])*lhs(1, N_bc)
        ybc = yupper + 0*xbc
        ubc = yupper_boundry_cond(tbc, ybc)

        x_bc_points.append(xbc), y_bc_points.append(ybc), u_bc_points.append(ubc), t_bc_points.append(tbc)


        x_bc_points = np.array(x_bc_points).flatten()
        t_bc_points = np.array(t_bc_points).flatten()
        u_bc_points = np.array(u_bc_points).flatten()
        u_bc_points = np.array(u_bc_points).flatten()
        u_bc_points = tf.reshape(u_bc_points, (N_bc*4,))

        # All boundary data
        bcs = np.column_stack([t_bc_points, x_bc_points,
                                u_bc_points]).astype(np.float64)
        
    return ["dirichlet_timeDependent", bcs, N_bc, xleft_boundry_cond, xright_boundry_cond, ylower_boundry_cond, yupper_boundry_cond]


""""User responsable for boundries to line up correctly"""
def setup_boundries_dirichlet_timeIndependent(x_bdry, y_bdry, N_bc=100, all_boundries_cond=None, xleft_boundry_cond=None, 
                                              xright_boundry_cond=None, ylower_boundry_cond=None, yupper_boundry_cond=None):
    xy_min = np.array([x_bdry[0], y_bdry[0]])
    xy_max = np.array([x_bdry[1], y_bdry[1]])
    xleft, xright = x_bdry[0], x_bdry[1]
    ylower, yupper = y_bdry[0], y_bdry[1]

    if all_boundries_cond != None:
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

""""User responsable for boundries to line up correctly"""
def setup_boundries_neumann_timeDependent(t_bdry, x_bdry, N_bc=100, all_boundries_cond=None, xleft_boundry_cond=None, 
                                            xright_boundry_cond=None):
    tx_min = np.array([t_bdry[0], x_bdry[0]])
    tx_max = np.array([t_bdry[1], x_bdry[1]])
    xleft, xright = x_bdry[0], x_bdry[1]

    if all_boundries_cond != None:
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
def setup_boundries_neumann_timeIndependent(x_bdry, y_bdry, N_bc=100, all_boundries_cond=None, xleft_boundry_cond=None, 
                                              xright_boundry_cond=None, ylower_boundry_cond=None, yupper_boundry_cond=None):

    xy_min = np.array([x_bdry[0], y_bdry[0]])
    xy_max = np.array([x_bdry[1], y_bdry[1]])
    xleft, xright = x_bdry[0], x_bdry[1]
    ylower, yupper = y_bdry[0], y_bdry[1]

    if all_boundries_cond != None:
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


# peridodic with inital conditions for all ut (x,t)
# periodic solely on all boudries (x,y)
#dirichlet with intial for all ut (x, t)
# dirichlet for all boundires (x, y)
