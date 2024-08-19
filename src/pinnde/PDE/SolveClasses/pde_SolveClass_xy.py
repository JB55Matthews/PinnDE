import numpy as np
import tensorflow as tf
import ast

from ..TrainingSelects import pde_trainingSelect_2var
from .. import pde_Points
from .. import pde_Plotters
from ..ModelFuncs import pde_ModelFuncs_2var as pde_ModelFuncs

""" Class containing ode solution information
    returned on solveODE calls
"""

class pde_xy_solution:
    
    def __init__(self, eqn, setup_boundaries, x_bdry, y_bdry, N_pde, net_layers, net_units, constraint, 
                 model, extra_ders, flag):
        """
        Constructer for class.

        Args:
            eqn (string): Equation to solve in form of string. function and derivatives represented as "u", "ux", "uy", "uxx", "uyy", 
                etc. for including function, i.e. cos(u), use tf.cos(u), or for ln(x), np.log(x). Write equation as would be written in code.
            setup_boundaries (boundary): boundary conditions set up from return of pde_Boundaries_2var call
            x_bdry (list): list of two elements, the interval of x to be solved on.
            y_bdry (list): list of two elements, the interval of y to be solved on.
            N_pde (int): Number of randomly sampled collocation points along x and y which PINN uses in training.
            net_layers (int): Number of internal layers of PINN
            net_units (int): Number of units in each internal layer
            constraint (string): Determines hard constrainting inital conditions or network learning inital conditions. "soft" or "hard"
            model (PINN): User may pass in user constructed network, however no guarentee of correct training.
            extra_ders (list): List of extra derivatives needed to be used. Network only computes single variable derivatives
                by default ("uxx", "uyyy", etc). If derivative not definded then input as string in list. Ex, if using
                "uxy" and "uxyx", then set extra_ders = ["uxy", "uxyx]
            flag (string): Internal string used to identify what problem is being solved

        Constructer also generates, model: model if model==None, pde_points: 
        randomly sampled points for model to train with, init_points: points along t0 for network to train with, X:
        sampled points along x direction, and Y: sampled points along y direction.
        Then calls PINNtrainSelect_xy function to train. Then defines solution prediction from returned model.

        Class Functions
        --------------
        Fucntions the user should call to access information from solvePDE call. Training, Getter, and Plotter functions

        Training
        ---------
        
        """

        self._eqn = eqn
        self._y_bdry = y_bdry
        self._x_bdry = x_bdry
        self._N_pde = N_pde
        #self._epochs = epochs
        self._epochs = 0
        self._flag = flag
        self._constraint = constraint
        self._net_layers = net_layers
        self._net_units = net_units
        self._setup_boundaries = setup_boundaries
        self._model = model
        self._extra_ders = extra_ders

        if model == None:
            self._model = pde_ModelFuncs.select_model_xy(x_bdry, y_bdry, net_layers, net_units, constraint, setup_boundaries)

        self._pde_points = pde_Points.defineCollocationPoints_2var(self._x_bdry, self._y_bdry, self._N_pde)

    
        # self._epoch_loss, self._iv_loss, self._bc_loss, self._pde_loss, self._model = pde_trainingSelect_2var.PINNtrainSelect_xy(self._pde_points, 
        # self._epochs, self._eqn, self._N_pde, self._setup_boundaries, self._model, self._constraint, self._extra_ders, self._flag)

        # # Grid where to evaluate the model
        # l, m = 100, 100
        # x = np.linspace(self._x_bdry[0], self._x_bdry[1], l)
        # y = np.linspace(self._y_bdry[0], self._y_bdry[1], m)
        # self._X, self._Y = np.meshgrid(x, y, indexing='ij')

        # self._solutionPred = self._model([np.expand_dims(self._X.flatten(), axis=1),
        #                         np.expand_dims(self._Y.flatten(), axis=1)])
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
        self._epoch_loss, self._iv_loss, self._bc_loss, self._pde_loss, self._model = pde_trainingSelect_2var.PINNtrainSelect_xy(self._pde_points, 
        self._epochs, self._eqn, self._N_pde, self._setup_boundaries, self._model, self._constraint, self._extra_ders, self._flag)

        # Grid where to evaluate the model
        l, m = 100, 100
        x = np.linspace(self._x_bdry[0], self._x_bdry[1], l)
        y = np.linspace(self._y_bdry[0], self._y_bdry[1], m)
        self._X, self._Y = np.meshgrid(x, y, indexing='ij')

        self._solutionPred = self._model([np.expand_dims(self._X.flatten(), axis=1),
                                np.expand_dims(self._Y.flatten(), axis=1)])
        self._solutionPred = np.reshape(self._solutionPred, (l, m))
        return

    def get_equation(self):
        """
        Returns:
            eqn (string): input equation
        """
        return self._eqn
    
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
    
    def get_bc_loss(self):
        """
        Return model boundary condition loss
        """
        return self._bc_loss
    
    def get_iv_loss(self):
        """
        Return model initial value loss
        """
        return self._iv_loss
    
    def get_x_points(self):
        """
        Return sampled x points
        """
        return self._X
    
    def get_y_points(self):
        """
        Return sampled y points
        """
        return self._Y
    
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
        Return sampled pe points

        Plotters
        -----------
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
    
    def plot_bc_loss(self, filetitle = "PDE-BC-Loss"):
        """
        Calls pde_Plotters.plot_bc_loss with correct data

        Args:
            filetitle (string): Title of saved file
        """
        pde_Plotters.plot_bc_loss(self._bc_loss, self._epochs, filetitle)
        return
    
    def plot_iv_loss(self, filetitle = "PDE-IV-Loss"):
        """
        Calls pde_Plotters.plot_iv_loss with correct data

        Args:
            filetitle (string): Title of saved file
        """
        pde_Plotters.plot_iv_loss(self._iv_loss, self._epochs, filetitle)
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
        pde_Plotters.plot_solution_prediction(self._X, self._Y, self._solutionPred, "xy", filetitle)
        return
    
    def plot_solution_prediction_3D(self, filetitle = "PDE-solution-Pred-3D"):
        """
        Calls pde_Plotters.plot_3D with correct data

        Args:
            filetitle (string): Title of saved file
        """
        pde_Plotters.plot_3D(self._X, self._Y, self._solutionPred, "xy", filetitle)
        return
    
    def plot_predicted_exact(self, exact_eqn, filetitle = "PDE-SolPred-Exact"):
        """
        Calls pde_Plotters.plot_predicted_exact with correct data

        Args:
            exact_eqn (string): Equation of exact solution which gets compared to model prediciton
            filetitle (string): Title of saved file
        """
        y = self._Y
        x = self._X
        parse_tree = ast.parse(exact_eqn, mode="eval")
        exact_eqn_data = eval(compile(parse_tree, "<string>", "eval"))
        pde_Plotters.plot_predicted_exact(self._X, self._Y, self._solutionPred, exact_eqn_data, 
                                          "xy", filetitle)
        return