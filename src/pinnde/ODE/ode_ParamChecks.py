

def solveODE_IVP_ParamCheck(eqn, order, init_data, t_bdry, N_pde, epochs, net_units, net_layers, constraint):
    """
    Parameter checking of solveODE_IVP calls. Raises error on incorrect input.

    Args:
        eqn (string): Should be equation to solve in form of string. function and derivatives represented as "u", "ut", "utt", 
            etc. for including function, i.e. cos(u), use tf.cos(u), or for ln(t), np.log(t). 
            Write equation as would be written in code.
        order (int): Should be order of equation (highest derivative used). Can be 1-5.
        init_data (list): Should be inital data for each deriviatve. Second order equation would have [u(t0), ut(t0)], 
            with t0 being inital t in t_bdry.
        t_bdry (list): Should be a list of two elements, the interval of t to be solved on.
        N_pde (int): Should be the number of randomly sampled collocation points along t which PINN uses in training.
        epochs (int): Should be the number of epochs PINN gets trained for.
        net_layers (int): Should be the number of internal layers of PINN
        net_units (int): Should be the number of units in each internal layer
        constraint (string): Should be string which determines hard constrainting inital conditions 
            or network learning inital conditions. "soft" or "hard"
    
    No Returns
    """

    if (type(N_pde) != int):
        raise TypeError("Number of selection points must be an integer")
    

    if (type(epochs) != int):
        raise TypeError("Number of training Epochs must be an integer")
    
    
    if (type(order) != int):
        raise TypeError("Order must be an integer")
    
    elif (order < 1 or order > 5):
        raise ValueError("Only order 1-5 equations supported")
    

    if (type(t_bdry) != list):
        raise TypeError("Boundries must be input as a list")
    
    elif (len(t_bdry) != 2):
        raise ValueError("Boundy list can only contain 2 elements")
    
    elif (t_bdry[1] < t_bdry[0]):
        raise ValueError("Left boundary must be less than right boundary")


    if (type(init_data) != list):
        raise TypeError("Initial conditions must be input as a list")
    
    elif (len(init_data) != order):
        raise TypeError("Must have same amount of inital conditions as order of equation to solve")
    
    if (type(net_units) != int):
        raise TypeError("Number of network units per layer must be an integer")
    
    if (type(net_layers) != int):
        raise TypeError("Number of network layers must be an integer")
    
    if ((constraint != "soft") and (constraint != "hard")):
        raise ValueError("Constraint must be \"soft\" or \"hard\"")
    
    #Eqn check not done, may just leave for trainign to catch

    return


def solveODE_BVP_ParamCheck(eqn, init_data, t_bdry, N_pde, epochs, order, net_units, net_layers, constraint):
    """
    Parameter checking of solveODE_BVP calls. Raises error on incorrect input.

    Args:
        eqn (string): Should be equation to solve in form of string. function and derivatives represented as "u", "ut", "utt", 
            etc. for including function, i.e. cos(u), use tf.cos(u), or for ln(t), np.log(t). 
            Write equation as would be written in code.
        init_data (list): Should be inital data for each deriviatve. Second order equation would have [u(t0), u(t1), ut(t0), u(t1)], 
            with t0 being inital t in t_bdry, t1 being final t in t_bdry.
        t_bdry (list): Should be a list of two elements, the interval of t to be solved on.
        N_pde (int): Should be the number of randomly sampled collocation points along t which PINN uses in training.
        epochs (int): Should be the number of epochs PINN gets trained for.
        order (int): Should be order of equation (highest derivative used). Can be 1-3.
        net_layers (int): Should be the number of internal layers of PINN
        net_units (int): Should be the number of units in each internal layer
        constraint (string): Should be string which determines hard constrainting inital conditions 
            or network learning inital conditions. "soft" or "hard"
    
    No Returns
    """

    if (type(N_pde) != int):
        raise TypeError("Number of selection points must be an integer")
    

    if (type(epochs) != int):
        raise TypeError("Number of training Epochs must be an integer")
    
    
    if (type(order) != int):
        raise TypeError("Order must be an integer")
    
    elif (order < 1 or order > 3):
        raise ValueError("Only order 1-3 equations supported")
    

    if (type(t_bdry) != list):
        raise TypeError("Boundries must be input as a list")
    
    elif (len(t_bdry) != 2):
        raise ValueError("Boundy list can only contain 2 elements")
    
    elif (t_bdry[1] < t_bdry[0]):
        raise ValueError("Left boundary must be less than right boundary")
    
    if (type(init_data) != list):
        raise TypeError("Initial conditions must be input as a list")
    
    elif (((order == 1) or (order == 2)) and (len(init_data) != 2)):
        raise ValueError("Initial boundary conditions must be [left boundary value, right boundary value]")
    
    elif ((order == 3) and (len(init_data) != 4)):
        raise ValueError("Initial boundary conditions must be [left boundary value, right boundary value, left boundary derivative value, right boundary derivative value]")

    if (type(net_units) != int):
        raise TypeError("Number of network units per layer must be an integer")
    
    if (type(net_layers) != int):
        raise TypeError("Number of network layers must be an integer")

    if ((constraint != "soft") and (constraint != "hard")):
        raise ValueError("Constraint must be \"soft\" or \"hard\"")

    return


def solveODESystem_IVP_ParamCheck(eqns, inits, t_bdry, N_pde, epochs, orders, net_layers, net_units, constraint):
    """
    Parameter checking of solveODE_BVP calls. Raises error on incorrect input.

    Args:
        eqns (string): Should be equations to solve in form of list of strings. function and derivatives represented as "u", "ut", "utt", 
            etc. for first equation. "x", "xt", etc. for second equation. "y", "yt", etc. for third equation.
            For including function, i.e. cos(u), use tf.cos(u), or for ln(t), np.log(t). Write equation as would be written in code.
        inits (list): Should be list of lists of inital data for each deriviatve. Previously descirbed orders would have
            [ [u(t0) ], [ x(t0), xt(t0), xtt(t0) ], [ y(t0), yt(t0) ]], with t0 being inital t in t_bdry.
        t_bdry (list): Should be a list of two elements, the interval of t to be solved on.
        N_pde (int): Should be the number of randomly sampled collocation points along t which PINN uses in training.
        epochs (int): Should be the number of epochs PINN gets trained for.
        orders (list): list of orders of equations (highest derivative used). Can be 1-3. ex. [1, 3, 2], corresponding to
            a highest derivative of "ut", "xttt", "ytt".
        net_layers (int): Should be the number of internal layers of PINN
        net_units (int): Should be the number of units in each internal layer
        constraint (string): Should be string which determines hard constrainting inital conditions 
            or network learning inital conditions. "soft" or "hard"
    
    No Returns
    """

    num_eqns = len(orders)

    if (type(N_pde) != int):
        raise TypeError("Number of selection points must be an integer")
    

    if (type(epochs) != int):
        raise TypeError("Number of training Epochs must be an integer")
    
    if (type(orders) != list):
        raise TypeError("Orders must be a list")
    
    elif(num_eqns != 2 and num_eqns != 3):
        raise ValueError("Must have orders for 2 or 3 equations")

    num_inits = 0
    for i in orders:
        if (i > 3 or i < 1):
            raise ValueError("Orders of equations must be from 1-3")
        else:
            num_inits += i

    if (type(t_bdry) != list):
        raise TypeError("Boundries must be input as a list")
    
    elif (len(t_bdry) != 2):
        raise ValueError("Boundy list can only contain 2 elements")
    
    elif (t_bdry[1] < t_bdry[0]):
        raise ValueError("Left boundary must be less than right boundary")
    
    if (len(inits) != num_inits):
        raise ValueError("Amount of inital conditions inconsistent with orders of equations given")
    
    if (type(net_units) != int):
        raise TypeError("Number of network units per layer must be an integer")
    
    if (type(net_layers) != int):
        raise TypeError("Number of network layers must be an integer")
    
    if ((constraint != "soft")):
        raise ValueError("Only soft constraints implemented currently")

    return

def solveODE_DeepONet_IVP_ParamCheck(eqn, order, init, t_bdry, N_pde, sensor_range, N_sensors, epochs, net_layers, 
                                     net_units, constraint):
    """
    Parameter checking of solveODE_IVP calls. Raises error on incorrect input.

    Args:
        eqn (string): Should be equation to solve in form of string. function and derivatives represented as "u", "ut", "utt", 
            etc. for including function, i.e. cos(u), use tf.cos(u), or for ln(t), np.log(t). 
            Write equation as would be written in code.
        order (int): Should be order of equation (highest derivative used). Can be 1-5.
        init (list): Should be inital data for each deriviatve. Second order equation would have [u(t0), ut(t0)], 
            with t0 being inital t in t_bdry.
        t_bdry (list): Should be a list of two elements, the interval of t to be solved on.
        N_pde (int): Should be the number of randomly sampled collocation points along t which DeepONet uses in training.
        sensor_range (list): Should be range in which sensors are sampled over.
        N_sensors (int): Should be number of sensors in which network learns over.
        epochs (int): Should be the number of epochs DeepONet gets trained for.
        net_layers (int): Should be the number of internal layers of DeepONet
        net_units (int): Should be the number of units in each internal layer
        constraint (string): Should be string which determines hard constrainting inital conditions 
            or network learning inital conditions. "soft" or "hard"
    
    No Returns
    """
    
    if (type(N_pde) != int):
        raise TypeError("Number of selection points must be an integer")
    

    if (type(epochs) != int):
        raise TypeError("Number of training Epochs must be an integer")
    
    
    if (type(order) != int):
        raise TypeError("Order must be an integer")
    
    elif (order < 1 or order > 3):
        raise ValueError("Only order 1-3 equations supported")
    

    if (type(t_bdry) != list):
        raise TypeError("Boundries must be input as a list")
    
    elif (len(t_bdry) != 2):
        raise ValueError("Boundy list can only contain 2 elements")
    
    elif (t_bdry[1] < t_bdry[0]):
        raise ValueError("Left boundary must be less than right boundary")


    if (type(init) != list):
        raise TypeError("Initial conditions must be input as a list")
    
    elif (len(init) != order):
        raise TypeError("Must have same amount of inital conditions as order of equation to solve")
    
    if ((constraint != "soft") and (constraint != "hard")):
        raise ValueError("Constraint must be \"soft\" or \"hard\"")
    
    if (type(sensor_range) != list):
        raise TypeError("Range for sensor points must be input as a list")
    elif (len(sensor_range) != 2):
        raise ValueError("sensor range list can only contain 2 elements, i.e, [-5, 5]")
    
    if (type(N_sensors) != int):
        raise TypeError("Number of sensors must be an integer")
    
    if (type(net_layers) != int):
        raise TypeError("Number of network layers must be an integer")
    
    if (type(net_units) != int):
        raise TypeError("Number of network units per layer must be an integer")

    return

def solveODE_DeepONet_BVP_ParamCheck(eqn, order, init, t_bdry, N_pde, sensor_range, N_sensors, epochs, net_layers, 
                                     net_units, constraint):
    """
    Parameter checking of solveODE_IVP calls. Raises error on incorrect input.

    Args:
        eqn (string): Should be equation to solve in form of string. function and derivatives represented as "u", "ut", "utt", 
            etc. for including function, i.e. cos(u), use tf.cos(u), or for ln(t), np.log(t). 
            Write equation as would be written in code.
        order (int): Should be order of equation (highest derivative used). Can be 1-5.
        init (list): Should be inital data for each deriviatve. Second order equation would have [u(t0), u(t1), ut(t0), u(t1)], 
            with t0 being inital t in t_bdry, t1 being final t in t_bdry.
        t_bdry (list): Should be a list of two elements, the interval of t to be solved on.
        N_pde (int): Should be the number of randomly sampled collocation points along t which DeepONet uses in training.
        sensor_range (list): Should be range in which sensors are sampled over.
        N_sensors (int): Should be number of sensors in which network learns over.
        epochs (int): Should be the number of epochs DeepONet gets trained for.
        net_layers (int): Should be the number of internal layers of DeepONet
        net_units (int): Should be the number of units in each internal layer
        constraint (string): Should be string which determines hard constrainting inital conditions 
            or network learning inital conditions. "soft" or "hard"
    
    No Returns
    """

    if (type(N_pde) != int):
        raise TypeError("Number of selection points must be an integer")
    

    if (type(epochs) != int):
        raise TypeError("Number of training Epochs must be an integer")
    
    
    if (type(order) != int):
        raise TypeError("Order must be an integer")
    
    elif (order < 1 or order > 3):
        raise ValueError("Only order 1-3 equations supported")
    

    if (type(t_bdry) != list):
        raise TypeError("Boundries must be input as a list")
    
    elif (len(t_bdry) != 2):
        raise ValueError("Boundy list can only contain 2 elements")
    
    elif (t_bdry[1] < t_bdry[0]):
        raise ValueError("Left boundary must be less than right boundary")


    if (type(init) != list):
        raise TypeError("Initial conditions must be input as a list")
    
    elif (((order == 1) or (order == 2)) and (len(init) != 2)):
        raise ValueError("Initial boundary conditions must be [left boundary value, right boundary value]")
    
    elif ((order == 3) and (len(init) != 4)):
        raise ValueError("Initial boundary conditions must be [left boundary value, right boundary value, left boundary derivative value, right boundary derivative value]")
    
    if ((constraint != "soft") and (constraint != "hard")):
        raise ValueError("Constraint must be \"soft\" or \"hard\"")
    
    if (type(sensor_range) != list):
        raise TypeError("Range for sensor points must be input as a list")
    elif (len(sensor_range) != 2):
        raise ValueError("sensor range list can only contain 2 elements, i.e, [-5, 5]")
    
    if (type(N_sensors) != int):
        raise TypeError("Number of sensors must be an integer")
    
    if (type(net_layers) != int):
        raise TypeError("Number of network layers must be an integer")
    
    if (type(net_units) != int):
        raise TypeError("Number of network units per layer must be an integer")

    return

def solveODE_DeepONetSystem_ParamCheck(eqns, orders, inits, t_bdry, N_pde, sensor_range, N_sensors, epochs, 
                        net_layers, net_units):
    """
    Parameter checking of solveODE_BVP calls. Raises error on incorrect input.

    Args:
        eqns (string): Should be equations to solve in form of list of strings. function and derivatives represented as "u", "ut", "utt", 
            etc. for first equation. "x", "xt", etc. for second equation. "y", "yt", etc. for third equation.
            For including function, i.e. cos(u), use tf.cos(u), or for ln(t), np.log(t). Write equation as would be written in code.
        inits (list): Should be list of lists of inital data for each deriviatve. Previously descirbed orders would have
            [ [u(t0) ], [ x(t0), xt(t0), xtt(t0) ], [ y(t0), yt(t0) ]], with t0 being inital t in t_bdry.
        t_bdry (list): Should be a list of two elements, the interval of t to be solved on.
        N_pde (int): Should be the number of randomly sampled collocation points along t which PINN uses in training.
        sensor_range (list): Should be range in which sensors are sampled over.
        N_sensors (int): Should be number of sensors in which network learns over.
        epochs (int): Should be the number of epochs DeepONet gets trained for.
        orders (list): list of orders of equations (highest derivative used). Can be 1-3. ex. [1, 3, 2], corresponding to
            a highest derivative of "ut", "xttt", "ytt".
        net_layers (int): Should be the number of internal layers of DeepONet
        net_units (int): Should be the number of units in each internal layer
    
    No Returns
    """

    num_eqns = len(orders)
    num_inits = 0
    for i in inits:
        for j in i:
            num_inits += 1

    if (type(N_pde) != int):
        raise TypeError("Number of selection points must be an integer")
    

    if (type(epochs) != int):
        raise TypeError("Number of training Epochs must be an integer")
    
    if (type(orders) != list):
        raise TypeError("Orders must be a list")
    
    elif(num_eqns != 2 and num_eqns != 3):
        raise ValueError("Must have orders for 2 or 3 equations")

    num_inits_should = 0
    for i in orders:
        if (i > 3 or i < 1):
            raise ValueError("Orders of equations must be from 1-3")
        else:
            num_inits_should += i

    if (type(t_bdry) != list):
        raise TypeError("Boundries must be input as a list")
    
    elif (len(t_bdry) != 2):
        raise ValueError("Boundy list can only contain 2 elements")
    
    elif (t_bdry[1] < t_bdry[0]):
        raise ValueError("Left boundary must be less than right boundary")
    
    if (num_inits != num_inits_should):
        raise ValueError("Amount of inital conditions inconsistent with orders of equations given")
    
    if (type(sensor_range) != list):
        raise TypeError("Range for sensor points must be input as a list")
    elif (len(sensor_range) != 2):
        raise ValueError("sensor range list can only contain 2 elements, i.e, [-5, 5]")
    
    if (type(N_sensors) != int):
        raise TypeError("Number of sensors must be an integer")
    
    if (type(net_layers) != int):
        raise TypeError("Number of network layers must be an integer")
    
    if (type(net_units) != int):
        raise TypeError("Number of network units per layer must be an integer")

    return