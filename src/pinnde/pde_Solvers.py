from .PDE.SolveClasses.pde_SolveClass_tx import pde_tx_solution
from .PDE.SolveClasses.pde_SolveClass_xy import pde_xy_solution
from .PDE.SolveClasses.pde_DeepONetSolveClass_tx import pde_don_tx_solution
from .PDE.SolveClasses.pde_DeepONetSolveClass_xy import pde_don_xy_solution
from .PDE.SolveClasses.pde_SolveClass_txy import pde_txy_solution
from .PDE import pde_ParamChecks

#Funnctions for solving ode
#main user functions


def solvePDE_tx(eqn, t_order, initial_t, setup_boundries, t_bdry=[0,1], x_bdry = [-1,1], N_pde=10000, N_iv = 100, epochs=1000, 
                  net_layers=4, net_units=60, constraint="soft", model=None, extra_ders = None):
    """
    Main function for solving inital condition boundary condition problem PDE in variables
        t and x with a PINN (physics informed neural network).

    Args:
        eqn (string): Equation to solve in form of string. Function and derivatives represented as "u", "ut", "ux", "utt", "uxx", 
            etc. for including function, i.e. cos(u), use tf.cos(u), or for ln(t), np.log(t). Write equation as would be written in code.
            **Must** be eqn = 0. Rearrange equation to equal 0.
        t_order (int): Order of t in equation (highest derivative of t used). Can be 1-3
        initial_t (lambda): Inital function for t=t0, as a python lambda funciton, with t0 being inital t in t_bdry.
            **Must** be of only 1 variable. See examples for how to make.
        setup_boundries (boundary): boundary conditions set up from return of pde_Boundries_2var call.
            See examples or API for boundries for how to use.
        t_bdry (list): List of two elements, the interval of t to be solved on.
        x_bdry (list): List of two elements, the interval of x to be solved on.
        N_pde (int): Number of randomly sampled collocation points along t and x which PINN uses in training.
        N_iv (int): Number of randomly sampled collocation points along inital t which PINN uses in training.
        epochs (int): Number of epochs PINN gets trained for.
        net_layers (int): Number of internal layers of PINN
        net_units (int): Number of units in each internal layer
        constraint (string): Determines hard constrainting inital conditions or network learning inital conditions. "soft" or "hard"
        model (PINN): User may pass in user constructed network, however no guarentee of correct training.
        extra_ders (list): List of extra derivatives needed to be used. Network only computes single variable derivatives
            by default ("utt", "uxxx", etc). If derivative not definded then input as string in list. Ex, if using
            "utx" and "utxt", then set extra_ders = ["utx", "utxt]

    Returns:
        solutionObj (pde_tx_solution): pde_tx_solution class containing all information from training

    """

    pde_ParamChecks.solvePDE_tx_ParamCheck(eqn, t_order, initial_t, setup_boundries, t_bdry, x_bdry, N_pde, N_iv, epochs,
                                             net_layers, net_units, model, constraint)

    solutionObj = pde_tx_solution(eqn, t_order, initial_t, setup_boundries, t_bdry, x_bdry, N_pde, N_iv, epochs, 
                                                 net_layers, net_units, constraint, model, extra_ders, "ICBCP-tx")
        
    return solutionObj

def solvePDE_xy(eqn, setup_boundries, x_bdry=[-1,1], y_bdry = [-1,1], N_pde=10000, epochs=1000, net_layers=4, net_units=60, 
                constraint="soft", model=None, extra_ders = None):
    """
    Main function for solving boundary condition problem PDE in variables
        x and y with a PINN (physics informed neural network).

    Args:
        eqn (string): Equation to solve in form of string. function and derivatives represented as "u", "ux", "uy", "uxx", "uyy", 
            etc. for including function, i.e. cos(u), use tf.cos(u), or for ln(x), np.log(x). Write equation as would be written in code.
            **Must** be eqn = 0. Rearrange equation to equal 0.
        setup_boundries (boundary): Boundary conditions set up from return of pde_Boundries_2var call.
            See examples or API for boundries for how to use.
        x_bdry (list): List of two elements, the interval of x to be solved on.
        y_bdry (list): List of two elements, the interval of y to be solved on.
        N_pde (int): Number of randomly sampled collocation points along x and y which PINN uses in training.
        epochs (int): Number of epochs PINN gets trained for.
        net_layers (int): Number of internal layers of PINN
        net_units (int): Number of units in each internal layer
        constraint (string): Determines hard constrainting inital conditions or network learning inital conditions. "soft" or "hard"
        model (PINN): User may pass in user constructed network, however no guarentee of correct training.
        extra_ders (list): List of extra derivatives needed to be used. Network only computes single variable derivatives
            by default ("uxx", "uyyy", etc). If derivative not definded then input as string in list. Ex, if using
            "uxy" and "uxyx", then set extra_ders = ["uxy", "uxyx]

    Returns:
        solutionObj (pde_xy_solution): pde_xy_solution class containing all information from training

    """
    pde_ParamChecks.solvePDE_xy_ParamCheck(eqn, setup_boundries, x_bdry, y_bdry, N_pde, epochs,
                                             net_layers, net_units, model, constraint)

    solutionObj = pde_xy_solution(eqn, setup_boundries, x_bdry, y_bdry, N_pde, epochs, 
                                                 net_layers, net_units, constraint, model, extra_ders, "BCP-xy")
        
    return solutionObj

def solvePDE_DeepONet_tx(eqn, t_order, initial_t, setup_boundries, t_bdry=[0,1], x_bdry = [-1,1], N_pde=10000, N_iv = 100, N_sensors=10000,
                           sensor_range=[-5,5], epochs=1000, net_layers=4, net_units=60, constraint="soft", extra_ders = None):
    """
    Main function for solving inital condition boundary condition problem PDE in variables
        t and x with a DeepONet (deep operator network).

    Args:
        eqn (string): Equation to solve in form of string. function and derivatives represented as "u", "ut", "ux", "utt", "uxx", 
            etc. for including function, i.e. cos(u), use tf.cos(u), or for ln(t), np.log(t). Write equation as would be written in code.
            **Must** be eqn = 0. Rearrange equation to equal 0.
        t_order (int): Order of t in equation (highest derivative of t used). Can be 1-3
        initial_t (lambda): Inital function for t=t0, as a python lambda funciton, with t0 being inital t in t_bdry.
            **Must** be of only 1 variable. See examples for how to make.
        setup_boundries (boundary): boundary conditions set up from return of pde_Boundries_2var call.
            See examples or API for boundries for how to use.
        t_bdry (list): List of two elements, the interval of t to be solved on.
        x_bdry (list): List of two elements, the interval of x to be solved on.
        N_pde (int): Number of randomly sampled collocation points to be used along t and x which DeepONet uses in training.
        N_iv (int): Number of randomly sampled collocation points to be used along inital t DeepONet uses in training
        N_sensors (int): Number of sensors in which network learns over. 
        sensor_range (list): Range in which sensors are sampled over.
        epochs (int): Number of epochs DeepONet gets trained for.
        net_layers (int): Number of internal layers of DeepONet
        net_units (int): Number of units in each internal layer
        constraint (string): Determines hard constrainting inital conditions or network learning inital conditions. "soft" or "hard"
        extra_ders (list): List of extra derivatives needed to be used. Network only computes single variable derivatives
            by default ("utt", "uxxx", etc). If derivative not definded then input as string in list. Ex, if using
            "utx" and "utxt", then set extra_ders = ["utx", "utxt]

    Returns:
        solutionObj (pde_don_tx_solution): pde_don_tx_solution class containing all information from training

    """

    pde_ParamChecks.solvePDE_DeepONet_tx_ParamCheck(eqn, t_order, initial_t, setup_boundries, t_bdry, x_bdry, N_pde, N_iv, N_sensors, 
                                        sensor_range, epochs, net_layers, net_units, constraint)

    solutionObj = pde_don_tx_solution(eqn, t_order, initial_t, setup_boundries, t_bdry, x_bdry, N_pde, N_iv, N_sensors, 
                                    sensor_range, epochs, net_layers, net_units, constraint, extra_ders, "DeepONet-ICBCP-tx")
        
    return solutionObj

def solvePDE_DeepONet_xy(eqn, setup_boundries, x_bdry=[-1,1], y_bdry = [-1,1], N_pde=10000, N_sensors=10000, 
                          sensor_range=[-5,5], epochs=1000, net_layers=4, net_units=60, constraint="soft", extra_ders = None):
    """
    Main function for solving boundary condition problem PDE in variables
        x and y with a DeepONet (deep operator network).

    Args:
        eqn (string): Equation to solve in form of string. function and derivatives represented as "u", "ux", "uy", "uxx", "uyy", 
            etc. for including function, i.e. cos(u), use tf.cos(u), or for ln(x), np.log(x). Write equation as would be written in code.
            **Must** be eqn = 0. Rearrange equation to equal 0.
        setup_boundries (boundary): Boundary conditions set up from return of pde_Boundries_2var call.
            See examples or API for boundries for how to use.
        x_bdry (list): List of two elements, the interval of x to be solved on.
        y_bdry (list): List of two elements, the interval of y to be solved on.
        N_pde (int): Number of randomly sampled collocation points to be used along x and y which DeepONet uses in training.
        N_sensors (int): Number of sensors in which network learns over. 
        sensor_range (list): Range in which sensors are sampled over.
        epochs (int): Number of epochs DeepONet gets trained for.
        net_layers (int): Number of internal layers of DeepONet
        net_units (int): Number of units in each internal layer
        constraint (string): Determines hard constrainting inital conditions or network learning inital conditions. "soft" or "hard"
        extra_ders (list): List of extra derivatives needed to be used. Network only computes single variable derivatives
            by default ("uxx", "uyyy", etc). If derivative not definded then input as string in list. Ex, if using
            "uxy" and "uxyx", then set extra_ders = ["uxy", "uxyx]
        
    Returns:
        solutionObj (pde_don_xy_solution): pde_don_xy_solution class containing all information from training

    """

    pde_ParamChecks.solvePDE_DeepONet_xy_ParamCheck(eqn, setup_boundries, x_bdry, y_bdry, N_pde, N_sensors, 
                                sensor_range, epochs, net_layers, net_units, constraint)

    solutionObj = pde_don_xy_solution(eqn, setup_boundries, x_bdry, y_bdry, N_pde, N_sensors, 
                                sensor_range, epochs, net_layers, net_units, constraint, extra_ders, "DeepONet-BCP-xy")
        
    return solutionObj


# def solvePDE_ICBCP_txy(eqn, t_order, initial_t, setup_boundries, t_bdry=[0,1], x_bdry = [-1,1], y_bdry = [-1,1], N_pde=10000, N_iv = 100, epochs=1000, 
#                   net_layers=4, net_units=60, model=None, constraint="soft"):
#     #Can take periodic dirichelt or vonneumann

#     # pde_ParamChecks.solvePDE_IBCP_ParamCheck(eqn, t_order, initial_t, setup_boundries, t_bdry, x_bdry, N_pde, N_iv, epochs,
#     #                                          net_layers, net_units, model, constraint)

#     solutionObj = pde_txy_solution(eqn, t_order, initial_t, setup_boundries, t_bdry, x_bdry, y_bdry, N_pde, N_iv, epochs, 
#                                                  net_layers, net_units, constraint, model, "ICBCP-txy")
        
#     return solutionObj