import ast
import numpy as np

from ..ode_trainingSelect import PINNtrainSelect_DeepONet
from .. import ode_TimeSteppersDeepONet 
from .. import ode_Plotters


class ode_DeepONetsolution:
    """
    Name of class returned
    """
    
    def __init__(self, eqn, inits, t_bdry, N_pde, N_sensors, sensor_range, epochs, order, net_layers, net_units, constraint, flag):
        """
        Constructer for class.

        Args:
            eqn (string): Equation to solve in form of string. function and derivatives represented as "u", "ut", "utt", 
                etc. for including function, i.e. cos(u), use tf.cos(u), or for ln(t), np.log(t). Write equation as would be written in code.
            inits (list): inital data for each deriviatve. Second order equation would have [u(t0), ut(t0)], 
                with t0 being inital t in t_bdry.
            t_bdry (list): list of two elements, the interval of t to be solved on.
            N_pde (int): Number of randomly sampled collocation points along t which DeepONet uses in training.
            N_sensors (int): Number of sensors in which network learns over.
            sensor_range (list): range in which sensors are sampled over.
            epochs (int): Number of epochs DeepONet gets trained for.
            order (int): order of equation (highest derivative used). Can be 1-3.
            net_layers (int): Number of internal layers of DeepONet
            net_units (int): Number of units in each internal layer
            constraint (string): Determines hard constrainting inital conditions or network learning inital conditions. "soft" or "hard"
            flag (string): Flag for internal use to identify which equation is being solved.
            
        Constructer calls PINNtrainSelect_DeepONet function to train, where all points are defined and solution prediction
        is generated within training file.

        Class Functions
        --------------
        Fucntions the user should call to access information from solveODE call. Getter functions, Plotter functions,
        and time stepping function

        Getters
        ----------

        """

        self._eqn = eqn
        self._inits = inits
        self._t_bdry = t_bdry
        self._N_pde = N_pde
        self._epochs = epochs
        self._order = order
        self._flag = flag
        self._N_sensors = N_sensors
        self._sensor_range = sensor_range
        self._net_layers = net_layers
        self._net_units = net_units
        self._constraint = constraint

        self._loss, self._t,self._solPred, self._params, self._model, self._de_points, self._sensors = PINNtrainSelect_DeepONet(self._N_pde, 
               self._inits, self._t_bdry, self._epochs, self._eqn, self._order, None, self._N_sensors, self._sensor_range,
              self._net_layers, self._net_units, self._constraint, self._flag)


    def get_equation(self):
        """
        Return input equation
        """
        return self._eqn
    
    def get_inits(self):
        """
        Return input inital data
        """
        return self._inits
    
    def get_t_bdry(self):
        """
        Return input t_bdry
        """
        return self._t_bdry
    
    def get_N_pde(self):
        """
        Return input N_pde
        """
        return self._N_pde
    
    def get_epochs(self):
        """
        Return input epochs
        """
        return self._epochs
    
    def get_order(self):
        """
        Return input order
        """
        return self._order
    
    def get_flag(self):
        """
        Return internal flag
        """
        return self._flag
    
    def get_N_sensors(self):
        """
        Return input N_sensors
        """
        return self._N_sensors
    
    def get_sensor_range(self):
        """
        Return input sensor range
        """
        return self._sensor_range
    
    def get_net_layers(self):
        """
        Return input net layers
        """
        return self._net_layers
    
    def get_net_units(self):
        """
        Return input net units
        """
        return self._net_units
    
    def get_constraint(self):
        """
        Return input constraint
        """
        return self._constraint
    
    def get_loss(self):
        """
        Return model total loss
        """
        return self._loss
    
    def get_t_points(self):
        """
        Return sampled t points
        """
        return self._t
    
    def get_solution_prediction(self):
        """
        Return model solution prediction
        """
        return self._solPred
    
    def get_params(self):
        """
        Return model paramaters
        """
        return self._params
    
    def get_model(self):
        """
        Return model
        """
        return self._model
    
    def get_de_points(self):
        """
        Return sampled de points
        """
        return self._de_points
    
    def get_sensors(self):
        """
        Return sensors

        Plotters
        --------
        """
        return self._sensors
    
    def plot_epoch_loss(self, filetitle = "ODE-Epoch-Loss"):
        """
        Calls ode_Plotters.plot_epoch_loss with correct data

        Args:
            filetitle (string): Title of saved file
        """
        ode_Plotters.plot_epoch_loss(self._loss, self._epochs, filetitle)
        return
    
    
    def plot_solution_prediction(self, filetitle = "ODE-solution-pred"):
        """
        Calls ode_Plotters.plot_solution_prediction with correct data

        Args:
            filetitle (string): Title of saved file
        """
        ode_Plotters.plot_solution_prediction(self._t, self._solPred, filetitle)
        return
    
    def plot_predicted_exact(self, exact_eqn, filetitle = "ODE-SolPred-Exact"):
        """
        Calls ode_Plotters.plot_predicted_exact with correct data

        Args:
            exact_eqn (lambda): Exact solution of equation as a python lambda function
            filetitle (string): Title of saved file

        Time Stepper
        ------------
        """
        exact_output = []
        for i in self._t:
            t = i
            parse_tree = ast.parse(exact_eqn, mode="eval")
            eqn = eval(compile(parse_tree, "<string>", "eval"))
            exact_output.append(eqn)
        ode_Plotters.plot_predicted_exact(self._t, self._solPred, exact_output, filetitle)
        return
    
    def timeStep(self, steps):
        """
        Calls ode_TimeSteppers function for corresponding equation type

        Args:
            steps (int): steps to timestep DeepONet. If original range was [0,2], and steps = 4, then timestep will be
                on [0, 8]
        """
        if self._flag == "DeepONetBVP":
            print("BVP cannot be time-stepped, must use an IVP with hard constraint for time stepping algorithm")
            return
        elif self._constraint == "soft":
            print("soft constrainct cannont be time-stepped,must use an IVP with hard constraint for time stepping algorithm ")
        else:
            if self._order == 1:
                return ode_TimeSteppersDeepONet.timeStep_order1(steps, self._t_bdry[1], self._model, self._params, self._N_pde, self._inits, self._t)
            elif self._order == 2:
                return ode_TimeSteppersDeepONet.timeStep_order2(steps, self._t_bdry[1], self._model, self._params, self._N_pde, self._inits, self._t)
            elif self._order == 3:
                return ode_TimeSteppersDeepONet.timeStep_order3(steps, self._t_bdry[1], self._model, self._params, self._N_pde, self._inits, self._t)
            return