from .ODE.SolveClasses.ode_SolveClass import ode_solution
from .ODE import ode_ParamChecks
from .ODE.SolveClasses.ode_SystemSolveClass import ode_systemSolution
from .ODE.SolveClasses.ode_DeepONetSolveClass import ode_DeepONetsolution
from .ODE.SolveClasses.ode_SystemDeepONetSolveClass import ode_SystemDeepONetsolution

#Funnctions for solving ode
#main user functions


def solveODE_IVP(eqn, order, init_data, t_bdry=[0,1], N_pde=100, epochs=1000, net_layers=4, net_units=40, constraint = "soft", model=None):
    """
    Main function for solving inital value problem ODE with a PINN (physics informed neural network).

    Args:
        eqn (string): Equation to solve in form of string. function and derivatives represented as "u", "ut", "utt", 
            etc. for including function, i.e. cos(u), use tf.cos(u), or for ln(t), np.log(t). Write equation as would be written in code.
            **Must** be eqn = 0. Rearrange equation to equal 0.
        order (int): order of equation (highest derivative used). Can be 1-5.
        init_data (list): inital data for each deriviatve. Second order equation would have [u(t0), ut(t0)], 
            with t0 being inital t in t_bdry.
        t_bdry (list): list of two elements, the interval of t to be solved on.
        N_pde (int): (Number points for differnetial equation). 
            Number of randomly sampled collocation points along t which PINN uses in training.
        epochs (int): Number of epochs PINN gets trained for.
        net_layers (int): Number of internal layers of PINN
        net_units (int): Number of units in each internal layer
        constraint (string): Determines hard constrainting inital conditions or network learning inital conditions. "soft" or "hard"
        model (PINN): User may pass in user constructed network, however no guarentee of correct training.

    Returns:
        solutionOBJ (ode_solution): ode_solution class containing all information from training

    """

    ode_ParamChecks.solveODE_IVP_ParamCheck(eqn, order, init_data, t_bdry, N_pde, epochs, net_units, net_layers, constraint)

    solutionObj = ode_solution(eqn, init_data, t_bdry, N_pde, epochs, order, net_layers, net_units, constraint, model, "IVP")
        
    return solutionObj

def solveODE_BVP(eqn, order, init_data, t_bdry=[0,1], N_pde=100, epochs=1000, net_layers=4, net_units=40, constraint = "soft", model=None):
    """
    Main function for solving boundary value problem ODE with a PINN (physics informed neural network).

    Args:
        eqn (string): Equation to solve in form of string. function and derivatives represented as "u", "ut", "utt", 
            etc. for including function, i.e. cos(u), use tf.cos(u), or for ln(t), np.log(t). Write equation as would be written in code.
            **Must** be eqn = 0. Rearrange equation to equal 0.
        order (int): order of equation (highest derivative used). Can be 1-3.
        init_data (list): inital boundary data for each deriviatve. First/second order equation would have [u(t0), u(t1)], Third order 
            equation would have [u(t0), u(t1), ut(t0), ut(t1)], with t0 being inital t in t_bdry, t1 being final t in t_bdry.
        t_bdry (list): list of two elements, the interval of t to be solved on.
        N_pde (int): (Number points for differnetial equation). 
            Number of randomly sampled collocation points along t which PINN uses in training.
        epochs (int): Number of epochs PINN gets trained for.
        net_layers (int): Number of internal layers of PINN
        net_units (int): Number of units in each internal layer
        constraint (string): Determines hard constrainting inital conditions or network learning inital conditions. "soft" or "hard"
        model (PINN): User may pass in user constructed network, however no guarentee of correct training.

    Returns:
        solutionOBJ (ode_solution): ode_solution class containing all information from training

    """

    ode_ParamChecks.solveODE_BVP_ParamCheck(eqn, init_data, t_bdry, N_pde, epochs, order, net_units, net_layers, constraint)

    solutionObj = ode_solution(eqn, init_data, t_bdry, N_pde, epochs, order, net_layers, net_units, constraint, model, "BVP")
        
    return solutionObj

def solveODE_System_IVP(eqns, orders, init_data, t_bdry=[0,1], N_pde=100, epochs=1000, net_layers=4, net_units=40, constraint = "soft", model=None):
    """
    Main function for solving systems of inital value problem ODE's with a PINN (physics informed neural network).

    Args:
        eqns (list): Equations to solve in form of list of strings. function and derivatives represented as "u", "ut", "utt", 
            etc. for first equation. "x", "xt", etc. for second equation. "y", "yt", etc. for third equation.
            For including function, i.e. cos(u), use tf.cos(u), or for ln(t), np.log(t). Write equation as would be written in code.
            **Must** be eqn = 0. Rearrange equation to equal 0.
        orders (list): list of orders of equations (highest derivative used). Can be 1-3. ex. [1, 3, 2], corresponding to
            a highest derivative of "ut", "xttt", "ytt".
        init_data (list): list of lists of inital data for each deriviatve. Previously descirbed orders would have
            [ [u(t0) ], [ x(t0), xt(t0), xtt(t0) ], [ y(t0), yt(t0) ]], with t0 being inital t in t_bdry.
        t_bdry (list): list of two elements, the interval of t to be solved on.
        N_pde (int): (Number points for differnetial equation). 
            Number of randomly sampled collocation points along t which PINN uses in training.
        epochs (int): Number of epochs PINN gets trained for.
        net_layers (int): Number of internal layers of PINN
        net_units (int): Number of units in each internal layer
        constraint (string): Determines hard constrainting inital conditions or network learning inital conditions. 
            Only "soft" implemented currently.
        model (PINN): User may pass in user constructed network, however no guarentee of correct training.

    Returns:
        solutionOBJ (ode_systemSolution): ode_systemSolution class containing all information from training

    """

    inits = []
    for i in init_data:
        for j in i:
            inits.append(j)

    ode_ParamChecks.solveODESystem_IVP_ParamCheck(eqns, inits, t_bdry, N_pde, epochs, orders, net_layers, net_units, constraint)

    solutionObj = ode_systemSolution(eqns, inits, t_bdry, N_pde, epochs, orders, net_layers, net_units, 
                                                          constraint, model, "System_IVP")
        
    return solutionObj

def solveODE_DeepONet_IVP(eqn, order, init, t_bdry=[0,1], N_pde=100, sensor_range = [-5, 5], N_sensors=3000, epochs=2000, 
                      net_layers = 4, net_units = 40, constraint = "hard"):
    """
    Main function for solving inital value problem ODE with a DeepONet (deep operator network).

    Args:
        eqn (string): Equation to solve in form of string. function and derivatives represented as "u", "ut", "utt", 
            etc. for including function, i.e. cos(u), use tf.cos(u), or for ln(t), np.log(t). Write equation as would be written in code.
            **Must** be eqn = 0. Rearrange equation to equal 0.
        order (int): order of equation (highest derivative used). Can be 1-3.
        init (list): inital data for each deriviatve. Second order equation would have [u(t0), ut(t0)], 
            with t0 being inital t in t_bdry.
        t_bdry (list): list of two elements, the interval of t to be solved on.
        N_pde (int): (Number points for differnetial equation). 
            Number of randomly sampled collocation points along t which DeepONet uses in training.
        sensor_range (list): range in which sensors are sampled over.
        N_sensors (int): Number of sensors in which network learns over.
        epochs (int): Number of epochs DeepONet gets trained for.
        net_layers (int): Number of internal layers of DeepONet
        net_units (int): Number of units in each internal layer
        constraint (string): Determines hard constrainting inital conditions or network learning inital conditions. "soft" or "hard"

    Returns:
        solutionOBJ (ode_DeepONetsolution): ode_DeepONetsolution class containing all information from training

    """

    ode_ParamChecks.solveODE_DeepONet_IVP_ParamCheck(eqn, order, init, t_bdry, N_pde, sensor_range, N_sensors, epochs,
                                                      net_layers, net_units, constraint)

    solutionObj = ode_DeepONetsolution(eqn, init, t_bdry, N_pde, N_sensors, sensor_range,
                                                               epochs, order, net_layers, net_units, constraint, "DeepONetIVP")

    return solutionObj

def solveODE_DeepONet_BVP(eqn, order, init, t_bdry=[0,1], N_pde=100, sensor_range = [-5, 5], N_sensors=3000, epochs=2000, 
                      net_layers = 4, net_units = 40, constraint="hard"):
    """
    Main function for solving boundary value problem ODE with a DeepONet (deep operator network).

    Args:
        eqn (string): Equation to solve in form of string. function and derivatives represented as "u", "ut", "utt", 
            etc. for including function, i.e. cos(u), use tf.cos(u), or for ln(t), np.log(t). Write equation as would be written in code.
            **Must** be eqn = 0. Rearrange equation to equal 0.
        order (int): order of equation (highest derivative used). Can be 1-3.
        init (list): inital boundary data for each deriviatve. First/second order equation would have [u(t0), u(t1)], Third order 
            equation would have [u(t0), u(t1), ut(t0), ut(t1)], with t0 being inital t in t_bdry, t1 being final t in t_bdry.
        t_bdry (list): list of two elements, the interval of t to be solved on.
        N_pde (int): (Number points for differnetial equation). 
            Number of randomly sampled collocation points along t which DeepONet uses in training.
        sensor_range (list): range in which sensors are sampled over.
        N_sensors (int): Number of sensors in which network learns over.
        epochs (int): Number of epochs DeepONet gets trained for.
        net_layers (int): Number of internal layers of DeepONet
        net_units (int): Number of units in each internal layer
        constraint (string): Determines hard constrainting inital conditions or network learning inital conditions. "soft" or "hard"

    Returns:
        solutionOBJ (ode_DeepONetsolution): ode_DeepONetsolution class containing all information from training

    """

    ode_ParamChecks.solveODE_DeepONet_BVP_ParamCheck(eqn, order, init, t_bdry, N_pde, sensor_range, N_sensors, epochs,
                                                      net_layers, net_units, constraint)

    solutionObj = ode_DeepONetsolution(eqn, init, t_bdry, N_pde, N_sensors, sensor_range,
                                                               epochs, order, net_layers, net_units, constraint, "DeepONetBVP")

    return solutionObj

def solveODE_DeepONetSystem_IVP(eqns, orders, init, t_bdry=[0,1], N_pde=100, sensor_range = [-5, 5], N_sensors=3000, epochs=2000, 
                      net_layers = 4, net_units = 40):
    """
    Main function for solving systems of inital value problem ODE's with a DeepONet (deep operator network).

    Args:
        eqns (list): Equations to solve in form of list of strings. function and derivatives represented as "u", "ut", "utt", 
            etc. for first equation. "x", "xt", etc. for second equation. "y", "yt", etc. for third equation.
            For including function, i.e. cos(u), use tf.cos(u), or for ln(t), np.log(t). Write equation as would be written in code.
            **Must** be eqn = 0. Rearrange equation to equal 0.
        orders (list): list of orders of equations (highest derivative used). Can be 1-3. ex. [1, 3, 2], corresponding to
            a highest derivative of "ut", "xttt", "ytt".
        init (list): list of lists of inital data for each deriviatve. Previously descirbed orders would have
            [[ u(t0) ], [ x(t0), xt(t0), xtt(t0) ], [ y(t0), yt(t0) ]], with t0 being inital t in t_bdry.
        t_bdry (list): list of two elements, the interval of t to be solved on.
        N_pde (int): (Number points for differnetial equation). 
            Number of randomly sampled collocation points along t which DeepONet uses in training.
        sensor_range (list): range in which sensors are sampled over.
        N_sensors (int): Number of sensors in which network learns over.
        epochs (int): Number of epochs DeepONet gets trained for.
        net_layers (int): Number of internal layers of DeepONet
        net_units (int): Number of units in each internal layer

    Returns:
        solutionOBJ (ode_SystemDeepONetsolution): ode_SystemDeepONetSolution class containing all information from training

    """
    
    ode_ParamChecks.solveODE_DeepONetSystem_ParamCheck(eqns, orders, init, t_bdry, N_pde, sensor_range, N_sensors, epochs, 
                        net_layers, net_units)

    solutionObj = ode_SystemDeepONetsolution(eqns, init, t_bdry, N_pde, N_sensors, sensor_range,
                                                               epochs, orders, net_layers, net_units, "DeepONetSys")

    return solutionObj

# def solveODE_HyperDeepONet_IVP(eqn, order, init, t_bdry=[0,1], N_pde=100, sensor_range = [-5, 5], N_sensors=3000, epochs=2000, 
#                       net_layers = 4, net_units = 40):

#     solutionObj = ode_DeepONetsolution(eqn, init, t_bdry, N_pde, N_sensors, sensor_range,
#                                                                epochs, order, net_layers, net_units, "HyperDeepONetIVP")

#     return solutionObj