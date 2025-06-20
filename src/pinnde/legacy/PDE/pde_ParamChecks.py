import inspect

def solvePDE_tx_ParamCheck(eqn, setup_initials, setup_boundaries, t_bdry, x_bdry, N_pde, epochs, 
                             net_layers, net_units, model, constraint):
    """
    Parameter checking of solvePDE_tx calls. Raises error on incorrect input.

    Args:
        eqn (string): Should be equation to solve in form of string. Function and derivatives represented as "u", "ut", "ux", "utt", "uxx", 
            etc. for including function, i.e. cos(u), use tf.cos(u), or for ln(t), np.log(t). Write equation as would be written in code.
        setup_initials (initial): initial conditions set up from return of pde_Initials.setup_initialconds_2var call.
            See examples or API for initials for how to use.
        setup_boundaries (boundary): Should be boundary conditions set up from return of pde_Boundaries_2var call
        t_bdry (list): Should be list of two elements, the interval of t to be solved on.
        x_bdry (list): Should be list of two elements, the interval of x to be solved on.
        N_pde (int): Should be number of randomly sampled collocation points along t and x which PINN uses in training.
        epochs (int): Should be number of epochs PINN gets trained for.
        net_layers (int): Should be number of internal layers of PINN
        net_units (int): Should be number of units in each internal layer
        model (PINN): User may pass in user constructed network, however no guarentee of correct training.
        constraint (string): Should be string which determines hard constrainting inital conditions or network learning inital conditions. "soft" or "hard"

    No returns

    **Note**: Hard constraints on dirichlet and neumann boundaries not implemented for equations in t and x

    """
    t_order = setup_initials[1]
    initial_t = setup_initials[2]
    N_iv = setup_initials[3]

    if (type(N_pde) != int):
        raise TypeError("Number of selection points must be an integer")
    
    if (type(N_iv) != int):
        raise TypeError("Number of selection points must be an integer")
    
    if (type(epochs) != int):
        raise TypeError("Number of training Epochs must be an integer")
    
    
    if (type(t_order) != int):
        raise TypeError("Order must be an integer")
    
    elif (t_order < 1 or t_order > 3):
        raise ValueError("Only t_order 1-3 equations supported")
    

    if (type(t_bdry) != list):
        raise TypeError("t_bdry - Boundaries must be input as a list")
    
    elif (len(t_bdry) != 2):
        raise ValueError("t_bdry - Boundy list can only contain 2 elements")
    
    elif (t_bdry[1] < t_bdry[0]):
        raise ValueError("t_bdry - Left boundary must be less than right boundary")
    
    if (type(x_bdry) != list):
        raise TypeError("x_bdry - Boundaries must be input as a list")
    
    elif (len(x_bdry) != 2):
        raise ValueError("x_bdry - Boundy list can only contain 2 elements")
    
    elif (x_bdry[1] < x_bdry[0]):
        raise ValueError("x_bdry - Left boundary must be less than right boundary")


    if (type(initial_t) != list):
        raise TypeError("Initial conditions must be input as a list")
    
    elif (len(initial_t) != t_order):
        raise TypeError("Must have same amount of inital conditions as order of equation to solve")
    
    for i in initial_t:
        args = (inspect.getfullargspec(i))[0]
        if (len(args) != 1):
            raise ValueError("Lambda functions for inital t must be functions of only 1 variable (x)")
    
    if (type(net_units) != int):
        raise TypeError("Number of network units per layer must be an integer")
    
    if (type(net_layers) != int):
        raise TypeError("Number of network layers must be an integer")
    
    if ((constraint != "soft") and (constraint != "hard")):
        raise ValueError("Constraint must be \"soft\" or \"hard\"")
    
    if setup_boundaries[0] != "periodic_timeDependent" and setup_boundaries[0] != "dirichlet_timeDependent" and setup_boundaries[0] !="neumann_timeDependent":
        raise ValueError("Boundary type not suitable with Initial Boundary Condtiion Problem, use: "
                         + "periodic_timeDependent, dirichlet_timeDependent, or neumann_timeDependent")
    
    #Not implemeneted errors:
    if constraint == "hard" and setup_boundaries[0] == "neumann_timeDependent":
        raise NotImplementedError("Hard constraints for Neumann boundaries on time dependent equations not implemented")
    if ((constraint == "hard") and (setup_boundaries[0] == "dirichlet_timeDependent") and (t_order != 1)):
        raise NotImplementedError("Hard constraints for Dirichlet boundaries on time dependent equations not implemented")
    
    #Eqn check not done, may just leave for trainign to catch

    return


def solvePDE_xy_ParamCheck(eqn, setup_boundaries, x_bdry, y_bdry, N_pde, epochs, net_layers, net_units, 
                 model, constraint):
    """
    Parameter checking of solvePDE_xy calls. Raises error on incorrect input.

    Args:
        eqn (string): Should be equation to solve in form of string. function and derivatives represented as "u", "ux", "uy", "uxx", "uyy", 
            etc. for including function, i.e. cos(u), use tf.cos(u), or for ln(x), np.log(x). Write equation as would be written in code.
        setup_boundaries (boundary): Should be boundary conditions set up from return of pde_Boundaries_2var call
        x_bdry (list): Should be list of two elements, the interval of x to be solved on.
        y_bdry (list): Should be list of two elements, the interval of y to be solved on.
        N_pde (int): Should be number of randomly sampled collocation points along x and y which PINN uses in training.
        epochs (int): Should be number of epochs PINN gets trained for.
        net_layers (int): Should be number of internal layers of PINN
        net_units (int): Should be number of units in each internal layer
        model (PINN): User may pass in user constructed network, however no guarentee of correct training.
        constraint (string): Should be string which determines hard constrainting inital conditions or network learning inital conditions. "soft" or "hard"

    No returns

    **Note**: Hard constraints on neumann boundaries not implemented for equations in x and y

    """

    if (type(N_pde) != int):
        raise TypeError("Number of selection points must be an integer")
    
    if (type(epochs) != int):
        raise TypeError("Number of training Epochs must be an integer")

    if (type(y_bdry) != list):
        raise TypeError("y_bdry - Boundaries must be input as a list")
    
    elif (len(y_bdry) != 2):
        raise ValueError("y_bdry - Boundy list can only contain 2 elements")
    
    elif (y_bdry[1] < y_bdry[0]):
        raise ValueError("y_bdry - Left boundary must be less than right boundary")
    
    if (type(x_bdry) != list):
        raise TypeError("x_bdry - Boundaries must be input as a list")
    
    elif (len(x_bdry) != 2):
        raise ValueError("x_bdry - Boundy list can only contain 2 elements")
    
    elif (x_bdry[1] < x_bdry[0]):
        raise ValueError("x_bdry - Left boundary must be less than right boundary")
    
    if (type(net_units) != int):
        raise TypeError("Number of network units per layer must be an integer")
    
    if (type(net_layers) != int):
        raise TypeError("Number of network layers must be an integer")
    
    if ((constraint != "soft") and (constraint != "hard")):
        raise ValueError("Constraint must be \"soft\" or \"hard\"")
    
    if setup_boundaries[0] != "periodic_timeIndependent" and setup_boundaries[0] != "dirichlet_timeIndependent" and setup_boundaries[0] !="neumann_timeIndependent":
        raise ValueError("Boundary type not suitable with Initial Boundary Condtiion Problem, use: "
                         + "periodic_timeIndependent, dirichlet_timeIndependent, or neumann_timeIndependent")
    
    #Not implemeneted errors:
    if constraint == "hard" and setup_boundaries[0] == "periodic_timeIndependent":
        print("Hard constraints for Periodic boundaries on time independent equations does not have an effect")
    if constraint == "hard" and setup_boundaries[0] == "neumann_timeIndependent":
        raise NotImplementedError("Hard constraints for Neumann boundaries on time dependent equations not implemented")
    
    #Eqn check not done, may just leave for trainign to catch

    return

def solvePDE_DeepONet_tx_ParamCheck(eqn, setup_initials, setup_boundaries, t_bdry, x_bdry, N_pde, N_sensors, 
                                        sensor_range, epochs, net_layers, net_units, constraint):
    """
    Parameter checking of solvePDE_DepONet_ty calls. Raises error on incorrect input.

    Args:
        eqn (string): Should be equation to solve in form of string. function and derivatives represented as "u", "ut", "ux", "utt", "uxx", 
            etc. for including function, i.e. cos(u), use tf.cos(u), or for ln(t), np.log(t). Write equation as would be written in code.
        setup_initials (initial): initial conditions set up from return of pde_Initials.setup_initialconds_2var call.
            See examples or API for initials for how to use.
        setup_boundaries (boundary): Should be boundary conditions set up from return of pde_Boundaries_2var call
        t_bdry (list): Should be list of two elements, the interval of t to be solved on.
        x_bdry (list): Should be list of two elements, the interval of x to be solved on.
        N_pde (int): Should be number of randomly sampled collocation points to be used along t and x which DeepONet uses in training.
        N_sensors (int): Should be number of sensors in which network learns over. 
        sensor_range (list): Should be range in which sensors are sampled over.
        epochs (int): Should be number of epochs DeepONet gets trained for.
        net_layers (int): Should be number of internal layers of DeepONet
        net_units (int): Should be number of units in each internal layer
        constraint (string): Should be string which determines hard constrainting inital conditions or network learning inital conditions. "soft" or "hard"

    No returns

    **Note**: Hard constraints on dirichlet and neumann boundaries not implemented for equations in t and x

    """

    t_order = setup_initials[1]
    initial_t = setup_initials[2]
    N_iv = setup_initials[3]
    
    if setup_boundaries[0] != "periodic_timeDependent":
        N_bc = setup_boundaries[2]
        if N_bc != N_iv:
            raise ValueError("Number of boundary condition points in boundaries must equal number of initial value points for network consistency")

    if (type(N_pde) != int):
        raise TypeError("Number of selection points must be an integer")
    
    if (type(N_sensors) != int):
        raise TypeError("Number of sensor points must be an integer")
    
    if (type(N_iv) != int):
        raise TypeError("Number of selection points must be an integer")
    
    if (type(epochs) != int):
        raise TypeError("Number of training Epochs must be an integer")
    
    
    if (type(t_order) != int):
        raise TypeError("Order must be an integer")
    
    elif (t_order < 1 or t_order > 3):
        raise ValueError("Only t_order 1-3 equations supported")
    
    if (type(sensor_range) != list):
        raise TypeError("Sensor range must be input as a list")
    
    elif (len(sensor_range) != 2):
        raise ValueError("Sensor range can only contain 2 elements")
    
    elif (sensor_range[1] < sensor_range[0]):
        raise ValueError("Sensor range left value must be less than right value")
    

    if (type(t_bdry) != list):
        raise TypeError("t_bdry - Boundaries must be input as a list")
    
    elif (len(t_bdry) != 2):
        raise ValueError("t_bdry - Boundy list can only contain 2 elements")
    
    elif (t_bdry[1] < t_bdry[0]):
        raise ValueError("t_bdry - Left boundary must be less than right boundary")
    
    if (type(x_bdry) != list):
        raise TypeError("x_bdry - Boundaries must be input as a list")
    
    elif (len(x_bdry) != 2):
        raise ValueError("x_bdry - Boundy list can only contain 2 elements")
    
    elif (x_bdry[1] < x_bdry[0]):
        raise ValueError("x_bdry - Left boundary must be less than right boundary")


    if (type(initial_t) != list):
        raise TypeError("Initial conditions must be input as a list")
    
    elif (len(initial_t) != t_order):
        raise TypeError("Must have same amount of inital conditions as order of equation to solve")
    
    for i in initial_t:
        args = (inspect.getfullargspec(i))[0]
        if (len(args) != 1):
            raise ValueError("Lambda functions for inital t must be functions of only 1 variable (x)")
    
    if (type(net_units) != int):
        raise TypeError("Number of network units per layer must be an integer")
    
    if (type(net_layers) != int):
        raise TypeError("Number of network layers must be an integer")
    
    if ((constraint != "soft") and (constraint != "hard")):
        raise ValueError("Constraint must be \"soft\" or \"hard\"")
    
    if setup_boundaries[0] != "periodic_timeDependent" and setup_boundaries[0] != "dirichlet_timeDependent" and setup_boundaries[0] !="neumann_timeDependent":
        raise ValueError("Boundary type not suitable with Initial Boundary Condtiion Problem, use: "
                         + "periodic_timeDependent, dirichlet_timeDependent, or neumann_timeDependent")
    
    #Not implemeneted errors:
    if constraint == "hard" and setup_boundaries[0] == "neumann_timeDependent":
        raise NotImplementedError("Hard constraints for Neumann boundaries on time dependent equations not implemented")
    if constraint == "hard" and setup_boundaries[0] == "dirichlet_timeDependent":
        raise NotImplementedError("Hard constraints for Dirichlet boundaries on time dependent equations not implemented")
    
    return

def solvePDE_DeepONet_xy_ParamCheck(eqn, setup_boundaries, x_bdry, y_bdry, N_pde, N_sensors, sensor_range, 
                            epochs, net_layers, net_units, constraint):
    """
    Parameter checking of solvePDE_DeepONet_xy calls. Raises error on incorrect input.

    Args:
        eqn (string): Should be equation to solve in form of string. function and derivatives represented as "u", "ux", "uy", "uxx", "uyy", 
            etc. for including function, i.e. cos(u), use tf.cos(u), or for ln(x), np.log(x). Write equation as would be written in code.
        setup_boundaries (boundary): Should be boundary conditions set up from return of pde_Boundaries_2var call
        x_bdry (list): Should be list of two elements, the interval of x to be solved on.
        y_bdry (list): Should be list of two elements, the interval of y to be solved on.
        N_pde (int): Should be number of randomly sampled collocation points to be used along x and y which DeepONet uses in training.
        N_sensors (int): Should be number of sensors in which network learns over. 
        sensor_range (list): Should be range in which sensors are sampled over.
        epochs (int): Should be number of epochs DeepONet gets trained for.
        net_layers (int): Should be number of internal layers of DeepONet
        net_units (int): Should be number of units in each internal layer
        constraint (string): Should be string which determines hard constrainting inital conditions or network learning inital conditions. "soft" or "hard"
        
    No returns

    **Note**: Hard constraints on neumann boundaries not implemented for equations in x and y

    """

    if (type(N_pde) != int):
        raise TypeError("Number of selection points must be an integer")
    
    if (type(N_sensors) != int):
        raise TypeError("Number of sensor points must be an integer")
    
    if (type(epochs) != int):
        raise TypeError("Number of training Epochs must be an integer")
    
    if (type(sensor_range) != list):
        raise TypeError("Sensor range must be input as a list")
    
    elif (len(sensor_range) != 2):
        raise ValueError("Sensor range can only contain 2 elements")
    
    elif (sensor_range[1] < sensor_range[0]):
        raise ValueError("Sensor range left value must be less than right value")

    if (type(y_bdry) != list):
        raise TypeError("y_bdry - Boundaries must be input as a list")
    
    elif (len(y_bdry) != 2):
        raise ValueError("y_bdry - Boundy list can only contain 2 elements")
    
    elif (y_bdry[1] < y_bdry[0]):
        raise ValueError("y_bdry - Left boundary must be less than right boundary")
    
    if (type(x_bdry) != list):
        raise TypeError("x_bdry - Boundaries must be input as a list")
    
    elif (len(x_bdry) != 2):
        raise ValueError("x_bdry - Boundy list can only contain 2 elements")
    
    elif (x_bdry[1] < x_bdry[0]):
        raise ValueError("x_bdry - Left boundary must be less than right boundary")
    
    if (type(net_units) != int):
        raise TypeError("Number of network units per layer must be an integer")
    
    if (type(net_layers) != int):
        raise TypeError("Number of network layers must be an integer")
    
    if ((constraint != "soft") and (constraint != "hard")):
        raise ValueError("Constraint must be \"soft\" or \"hard\"")
    
    if setup_boundaries[0] != "periodic_timeIndependent" and setup_boundaries[0] != "dirichlet_timeIndependent" and setup_boundaries[0] !="neumann_timeIndependent":
        raise ValueError("Boundary type not suitable with Initial Boundary Condtiion Problem, use: "
                         + "periodic_timeIndependent, dirichlet_timeIndependent, or neumann_timeIndependent")
    
    if setup_boundaries[2] != N_pde:
        return
    
    #Not implemeneted errors:
    if constraint == "hard" and setup_boundaries[0] == "periodic_timeIndependent":
        print("Hard constraints for Periodic boundaries on time independent equations does not have an effect")
    if constraint == "hard" and setup_boundaries[0] == "neumann_timeIndependent":
        raise NotImplementedError("Hard constraints for Neumann boundaries on time dependent equations not implemented")
    
    return