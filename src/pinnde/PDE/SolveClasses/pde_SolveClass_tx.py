import numpy as np
import tensorflow as tf
import ast

from ..pde_Points import defineCollocationPoints_2var
from .. import pde_Plotters
from ..ModelFuncs.pde_ModelFuncs_2var import select_model_tx
from ..TrainingSelects.pde_trainingSelect_2var import PINNtrainSelect_tx


class pde_tx_solution:
    """
    Name of class returned
    """
    
    def __init__(self, eqn, setup_initials, setup_boundaries, t_bdry, x_bdry, N_pde,
                 net_layers, net_units, constraint, model, extra_ders, flag):
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
            N_pde (int): Number of randomly sampled collocation points along t and x which PINN uses in training.
            net_layers (int): Number of internal layers of PINN
            net_units (int): Number of units in each internal layer
            constraint (string): Determines hard constrainting inital conditions or network learning inital conditions. "soft" or "hard"
            model (PINN): User may pass in user constructed network, however no guarentee of correct training.
            extra_ders (list): List of extra derivatives needed to be used. Network only computes single variable derivatives
                by default ("utt", "uxxx", etc). If derivative not definded then input as string in list. Ex, if using
                "utx" and "utxt", then set extra_ders = ["utx", "utxt]
            flag (string): Internal string used to identify what problem is being solved

        Constructer also generates, model: model if model==None, pde_points: 
        randomly sampled points for model to train with, init_points: points along t0 for network to train with, T:
        sampled points along t direction, and X: sampled points along x direction.
        Then train_model should be call to call PINNtrainSelect_tx function to train, 
        then defines solution prediction from returned model.

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
        #self._epochs = epochs
        self._epochs = 0
        self._t_order = setup_initials[1]
        self._flag = flag
        self._constraint = constraint
        self._net_layers = net_layers
        self._net_units = net_units
        self._model = model
        self._extra_ders = extra_ders

        self._init_points = setup_initials[0]

        if self._model == None:
            self._model = select_model_tx(t_bdry, x_bdry, self._t_order, self._initial_t, net_layers, net_units, constraint, setup_boundaries)
        
        self._pde_points = defineCollocationPoints_2var(self._t_bdry, self._x_bdry, self._N_pde)

        # self._epoch_loss, self._iv_loss, self._bc_loss, self._pde_loss, self._model = PINNtrainSelect_tx(self._pde_points, 
        # self._init_points, self._epochs, self._eqn, self._t_order, self._N_pde,
        # self._N_iv, self._setup_boundaries, self._model, self._constraint, self._extra_ders, self._flag)

        # # Grid where to evaluate the model
        # l, m = 100, 1000
        # self._t = np.linspace(self._t_bdry[0], self._t_bdry[1], l)
        # self._x = np.linspace(self._x_bdry[0], self._x_bdry[1], m)
        # self._T, self._X = np.meshgrid(self._t, self._x, indexing='ij')

        # self._solutionPred = self._model([np.expand_dims(self._T.flatten(), axis=1),
        #         np.expand_dims(self._X.flatten(), axis=1)])
        # self._solutionPred = np.reshape(self._solutionPred, (l, m))

    def train_model(self, epochs):
        """
        Main function to train model. Defines solution prediction from model, and generates meshgrid data for plotting

        Args:
            epochs (int): Epochs to train model for to epoch train

        Getters
        ---------

        """
        self._epochs = epochs
        self._epoch_loss, self._iv_loss, self._bc_loss, self._pde_loss, self._model = PINNtrainSelect_tx(self._pde_points, 
        self._init_points, self._epochs, self._eqn, self._t_order, self._N_pde,
        self._N_iv, self._setup_boundaries, self._model, self._constraint, self._extra_ders, self._flag)

        # Grid where to evaluate the model
        l, m = 100, 1000
        self._t = np.linspace(self._t_bdry[0], self._t_bdry[1], l)
        self._x = np.linspace(self._x_bdry[0], self._x_bdry[1], m)
        self._T, self._X = np.meshgrid(self._t, self._x, indexing='ij')

        self._solutionPred = self._model([np.expand_dims(self._T.flatten(), axis=1),
                np.expand_dims(self._X.flatten(), axis=1)])
        self._solutionPred = np.reshape(self._solutionPred, (l, m))
        return

    def get_equation(self):
        """
        Return input equation
        """
        return self._eqn
    
    def get_initial_t(self):
        """
        Return input initial_t function
        """
        return self._initial_t
    
    def get_t_bdry(self):
        """
        Return input t_bdry
        """
        return self._t_bdry
    
    def get_x_bdry(self):
        """
        Return input x_bdry
        """
        return self._x_bdry
    
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
        Return sampled t points
        """
        return self._T
    
    def get_x_points(self):
        """
        Return sampled x points
        """
        return self._X
    
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

        Plotters
        ----------
        """
        return self._pde_points
    
    
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
        Calls pde_Plotters.plot_all_losse with correct data

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
        Calls pde_Plotters.plot_3d with correct data

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