import numpy as np
import tensorflow as tf
import ast

from ..pde_Points import defineCollocationPoints_DON_2var
from .. import pde_Plotters
from ..ModelFuncs.pde_DeepONetModelFuncs_2var import select_DeepONet_tx
from ..TrainingSelects.pde_trainingSelect_2var import PINNtrainSelect_DeepONet_tx
from .. import pde_TimeSteppersDeepONet

class pde_don_tx_solution:
    """
    Name of class returned
    """
    
    def __init__(self, eqn, setup_initials, setup_boundaries, t_bdry, x_bdry, N_pde, N_sensors, 
                 sensor_range, net_layers, net_units, constraint, extra_ders, flag):
        """
        Constructer for class.

        Args:
            eqn (string): Equation to solve in form of string. function and derivatives represented as "u", "ut", "ux", "utt", "uxx", 
                etc. for including function, i.e. cos(u), use tf.cos(u), or for ln(t), np.log(t). Write equation as would be written in code.
            setup_initials (initial): initial conditions set up from return of pde_Initials.setup_initialconds_2var call.
                See examples or API for initials for how to use.
            setup_boundaries (boundary): boundary conditions set up from return of pde_Boundaries_2var call
            t_bdry (list): list of two elements, the interval of t to be solved on.
            x_bdry (list): list of two elements, the interval of x to be solved on.
            N_pde (int): Number of randomly sampled collocation points to be used along t and x which DeepONet uses in training.
            N_sensors (int): Number of sensors in which network learns over. 
            sensor_range (list): range in which sensors are sampled over.
            net_layers (int): Number of internal layers of DeepONet
            net_units (int): Number of units in each internal layer
            constraint (string): Determines hard constrainting inital conditions or network learning inital conditions. "soft" or "hard"
            extra_ders (list): List of extra derivatives needed to be used. Network only computes single variable derivatives
                by default ("utt", "uxxx", etc). If derivative not definded then input as string in list. Ex, if using
                "utx" and "utxt", then set extra_ders = ["utx", "utxt]
            flag (string): Internal string used to identify what problem is being solved

        Constructer also generates, model: model if model==None, pde_points: 
        randomly sampled points for model to train with, init_points: points along t0 for network to train with, T:
        sampled points along t direction, and X: sampled points along x direction.
        Then calls PINNtrainSelect_DeepONet_tx function to train. Then defines solution prediction from returned model.

        Class Functions
        --------------
        Fucntions the user should call to access information from solvePDE call. Training, Getter, and Plotter functions

        Training
        ---------
        
        """

        self._eqn = eqn
        self._setup_boundaries = setup_boundaries
        self._initial_t = setup_initials[2]
        self._t_bdry = t_bdry
        self._x_bdry = x_bdry
        self._N_pde = N_pde
        self._N_iv = setup_initials[3]
        self._epochs = 0
        self._t_order = setup_initials[1]
        self._flag = flag
        self._constraint = constraint
        self._net_layers = net_layers
        self._net_units = net_units
        self._sensor_range = sensor_range
        self._N_sensors = N_sensors
        self._extra_ders = extra_ders

        self._init_points = setup_initials[0]

        self._model = select_DeepONet_tx(t_bdry, x_bdry, self._t_order, self._initial_t, net_layers, net_units, 
                                                                constraint, setup_boundaries, self._N_iv)
        
        self._pde_points, self._usensors = defineCollocationPoints_DON_2var(t_bdry, x_bdry,
                                                                                N_pde, self._N_iv, N_sensors, sensor_range)

        # self._epoch_loss, self._iv_loss, self._bc_loss, self._pde_loss, self._model = PINNtrainSelect_DeepONet_tx(self._pde_points, 
        # self._init_points, self._epochs, self._eqn, self._t_order, self._N_pde, self._N_iv, self._N_sensors, self._usensors,
        # self._sensor_range, self._setup_boundaries, self._model, self._constraint, self._extra_ders, self._flag)

        # # Grid where to evaluate the model
        # def factors(x):
        #     return [i for i in range(1,x+1) if x%i==0]
        # N_sensor_facts = factors(N_sensors)
        # if len(N_sensor_facts) % 2 == 0:
        #     self._l = N_sensor_facts[len(N_sensor_facts)//2-1]
        #     self._m = N_sensor_facts[len(N_sensor_facts)//2]
        # else:
        #     self._l = self._m = N_sensor_facts[(len(N_sensor_facts)-1)//2]
        # t = np.linspace(self._t_bdry[0], self._t_bdry[1], self._l)
        # x = np.linspace(self._x_bdry[0], self._x_bdry[1], self._m)
        # self._T, self._X = np.meshgrid(t, x, indexing='ij')
              
        # self._solutionPred = self._model([np.expand_dims(self._T.flatten(), axis=1),
        #         np.expand_dims(self._X.flatten(), axis=1), self._usensors])
        # self._solutionPred = np.reshape(self._solutionPred, (self._l, self._m))

    def train_model(self, epochs):
        """
        Main function to train model. Defines solution prediction from model, and generates meshgrid data for plotting

        Args:
            epochs (int): Epochs to train model for to epoch train

        Getters
        ---------

        """
        self._epochs = epochs
        self._epoch_loss, self._iv_loss, self._bc_loss, self._pde_loss, self._model = PINNtrainSelect_DeepONet_tx(self._pde_points, 
        self._init_points, self._epochs, self._eqn, self._t_order, self._N_pde, self._N_iv, self._N_sensors, self._usensors,
        self._sensor_range, self._setup_boundaries, self._model, self._constraint, self._extra_ders, self._flag)

        # Grid where to evaluate the model
        def factors(x):
            return [i for i in range(1,x+1) if x%i==0]
        N_sensor_facts = factors(self._N_sensors)
        if len(N_sensor_facts) % 2 == 0:
            self._l = N_sensor_facts[len(N_sensor_facts)//2-1]
            self._m = N_sensor_facts[len(N_sensor_facts)//2]
        else:
            self._l = self._m = N_sensor_facts[(len(N_sensor_facts)-1)//2]
        t = np.linspace(self._t_bdry[0], self._t_bdry[1], self._l)
        x = np.linspace(self._x_bdry[0], self._x_bdry[1], self._m)
        self._T, self._X = np.meshgrid(t, x, indexing='ij')
              
        self._solutionPred = self._model([np.expand_dims(self._T.flatten(), axis=1),
                np.expand_dims(self._X.flatten(), axis=1), self._usensors])
        self._solutionPred = np.reshape(self._solutionPred, (self._l, self._m))
        return

    def get_equation(self):
        """
        Return input equation
        """
        return self._eqn
    
    def get_initial_t(self):
        """
        Return input initial_t
        """
        return self._initial_t
    
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
    
    def get_t_order(self):
        """
        Return input t_order
        """
        return self._t_order
    
    def get_flag(self):
        """
        Return internal flag
        """
        return self._flag
    
    def get_constraint(self):
        """
        Return input constraint
        """
        return self._constraint
    
    def get_epoch_loss(self):
        """
        Return model total epoch loss
        """
        return self._epoch_loss
    
    def get_pde_loss(self):
        """
        Return model pde loss
        """
        return self._pde_loss
    
    def get_iv_loss(self):
        """
        Return model initial value loss
        """
        return self._iv_loss
    
    def get_bc_loss(self):
        """
        Return model boundary condition loss
        """
        return self._bc_loss
    
    def get_t_points(self):
        """
        Return sampled points along t
        """
        return self._T
    
    def get_x_points(self):
        """
        Return sampled points along x
        """
        return self._X
    
    def get_usensors(self):
        """
        Return sampled sensors for u
        """
        return self._usensors
    
    def get_solution_prediction(self):
        """
        Return model solution prediction
        """
        return self._solutionPred
    
    def get_model(self):
        """
        Return trained model
        """
        return self._model
    
    def get_pde_points(self):
        """
        Return sampled pde points
        """
        return self._pde_points
    
    def get_N_sensors(self):
        """
        Return input N_sensors
        """
        return self._N_sensors
    
    def get_sensor_range(self):
        """
        Return input sensor range

        Plotters
        ---------
        """
        return self._sensor_range
    
    
    def plot_epoch_loss(self, filetitle = "PDE-Epoch-Loss"):
        """
        Calls pde_Plotters.plot_epoch_loss with correct data

        Args:
            filetitle (string): Title of saved file
        """
        pde_Plotters.plot_epoch_loss(self._epoch_loss, self._epochs, filetitle)
        return
    
    def plot_iv_loss(self, filetitle = "PDE-IV-Loss"):
        """
        Calls pde_Plotters.plot_iv_loss with correct data

        Args:
            filetitle (string): Title of saved file
        """
        pde_Plotters.plot_iv_loss(self._iv_loss, self._epochs, filetitle)
        return
    
    def plot_bc_loss(self, filetitle = "PDE-BC-Loss"):
        """
        Calls pde_Plotters.plot_bc_loss with correct data

        Args:
            filetitle (string): Title of saved file
        """
        pde_Plotters.plot_bc_loss(self._bc_loss, self._epochs, filetitle)
        return

    def plot_pde_loss(self, filetitle = "PDE-DE-Loss"):
        """
        Calls pde_Plotters.plot_pde_loss with correct data

        Args:
            filetitle (string): Title of saved file
        """
        pde_Plotters.plot_pde_loss(self._pde_loss, self._epochs, filetitle)
        return
    
    def plot_all_losses(self, filetitle = "PDE-All-Losses"):
        """
        Calls pde_Plotters.plot_all_losses with correct data

        Args:
            filetitle (string): Title of saved file
        """
        pde_Plotters.plot_all_losses(self._epoch_loss, self._pde_loss, self._iv_loss, self._bc_loss, self._epochs, filetitle)
    
    def plot_solution_prediction(self, filetitle = "PDE-solution-pred"):
        """
        Calls pde_Plotters.plot_solution_prediction with correct data

        Args:
            filetitle (string): Title of saved file
        """
        pde_Plotters.plot_solution_prediction(self._T, self._X, self._solutionPred, "tx", filetitle)
        return
    
    def plot_solution_prediction_3D(self, filetitle = "PDE-solution-Pred-3D"):
        """
        Calls pde_Plotters.plot_3D with correct data

        Args:
            filetitle (string): Title of saved file
        """
        pde_Plotters.plot_3D(self._T, self._X, self._solutionPred, "tx", filetitle)
        return
    
    def plot_predicted_exact(self, exact_eqn, filetitle = "PDE-SolPred-Exact"):
        """
        Calls pde_Plotters.plot_predicted_exact with correct data

        Args:
            exact_eqn (string): Equation of exact solution which gets compared to model prediciton
            filetitle (string): Title of saved file
        """
        t = self._T
        x = self._X
        parse_tree = ast.parse(exact_eqn, mode="eval")
        exact_eqn_data = eval(compile(parse_tree, "<string>", "eval"))
        pde_Plotters.plot_predicted_exact(self._T, self._X, self._solutionPred, exact_eqn_data, 
                                          "tx", filetitle)
        return
    
    def timeStep(self, steps):
        return pde_TimeSteppersDeepONet.timeStep_ordert1(steps, self._t_bdry, self._model, self._initial_t[0], self._T, self._X, 
                                        self._x_bdry, self._N_iv)
    