import ast
import numpy as np

from ..ode_trainingSelect import PINNtrainSelect_DeepONet
from .. import ode_TimeSteppersDeepONet
from .. import ode_Plotters


class ode_SystemDeepONetsolution:
    """
    Name of class returned
    """

    def __init__(self, eqns, inits, t_bdry, N_pde, N_sensors, sensor_range, epochs, order, net_layers, net_units, flag):
        """
        Constructer for class.

        Args:
            eqns (list): Equations to solve in form of list of strings. function and derivatives represented as "u", "ut", "utt", 
                etc. for first equation. "x", "xt", etc. for second equation. "y", "yt", etc. for third equation.
                For including function, i.e. cos(u), use tf.cos(u), or for ln(t), np.log(t). Write equation as would be written in code.
            inits (list): list of lists of inital data for each deriviatve. Previously descirbed orders would have
                [[ u(t0) ], [ x(t0), xt(t0), xtt(t0) ], [ y(t0), yt(t0) ]], with t0 being inital t in t_bdry.
            t_bdry (list): list of two elements, the interval of t to be solved on.
            N_pde (int): Number of randomly sampled collocation points along t which DeepONet uses in training.
            N_sensors (int): Number of sensors in which network learns over.
            sensor_range (list): range in which sensors are sampled over.
            epochs (int): Number of epochs DeepONet gets trained for.
            order (list): list of orders of equations (highest derivative used). Can be 1-3. ex. [1, 3, 2], corresponding to
                a highest derivative of "ut", "xttt", "ytt".
            net_layers (int): Number of internal layers of DeepONet
            net_units (int): Number of units in each internal layer
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

        self._original_eqns = list(eqns)
        self._original_orders = list(order)
        self._t_bdry = t_bdry
        self._N_pde = N_pde
        self._epochs = epochs
        self._flag = flag
        self._N_sensors = N_sensors
        self._sensor_range = sensor_range
        self._net_layers = net_layers
        self._net_units = net_units
        
        self._eqns, self._inits, self._order = self.__reOrder_eqns(eqns, inits, order)

        self._loss, self._t, solPredTemp, self._params, self._model, self._de_points, self._sensors = PINNtrainSelect_DeepONet(self._N_pde, 
               self._inits, self._t_bdry, self._epochs, self._eqns, self._order, self._original_orders, self._N_sensors, 
               self._sensor_range, self._net_layers, self._net_units, None, self._flag)
        
        self._trainedSolPred = list(solPredTemp)
        self._solPred = self.__reOrder_solPred(solPredTemp, self._original_eqns, self._eqns)


    def get_equations(self):
        """
        Return input equation
        """
        return self._eqns
    
    def get_inits(self):
        """
        Return input initial data
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
        Return input orders
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
        Return input sensor_range
        """
        return self._sensor_range
    
    def get_net_layers(self):
        """
        Return input net_layers
        """
        return self._net_layers
    
    def get_net_units(self):
        """
        Return input net_units
        """
        return self._net_units
    
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
        Return model parameters
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
        ----------
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
        ode_Plotters.plot_solution_prediction_system(self._t, self._solPred, filetitle)
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
        exact_outs = []
        for j in range(len(self._order)):
            exact_output = []
            for i in self._t:
                t = i
                parse_tree = ast.parse(exact_eqn[j], mode="eval")
                eqn = eval(compile(parse_tree, "<string>", "eval"))
                exact_output.append(eqn)
            exact_outs.append(exact_output)
        ode_Plotters.plot_predicted_exact_system(self._t, self._solPred, exact_outs, filetitle)
        return
    
    def timeStep(self, steps, filetitle = "ODE-TimeStep-Pred"):
        """
        Calls ode_TimeSteppers function for corresponding equation type

        Args:
            steps (int): steps to timestep DeepONet. If original range was [0,2], and steps = 4, then timestep will be
                on [0, 8]
            filetitle (string): Title of saved file 
        """
        
        return ode_TimeSteppersDeepONet.timeStep_SystemSelect(steps, self._t_bdry[1], self._model, self._params, 
                                        self._N_pde, self._inits, self._t, self._order, self._original_orders, filetitle)
    
    def __reOrder_eqns(self, eqns, inits, order):
    
        return_eqns = []
        return_inits = []
        return_orders = []
        while len(order) != 0:
            max_order = order[0]
            index = 0
            for i in range(len(order)):
                if order[i] > max_order:
                    max_order = order[i]
                    index = i
            return_eqns.append(eqns.pop(index))
            return_inits.append(inits.pop(index))
            return_orders.append(order.pop(index))
        return return_eqns, return_inits, return_orders
    
    def __reOrder_solPred(self, solPredTemp, original_eqns, eqns):
        return_solPred = []
        for i in range(len(eqns)):
            index = eqns.index(original_eqns[i])
            return_solPred.insert(i, solPredTemp[index])
        return return_solPred