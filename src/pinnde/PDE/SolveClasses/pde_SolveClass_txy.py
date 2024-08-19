import numpy as np
import math

from ..pde_Points import defineCollocationPoints_txy
from .. import pde_Plotters
from ..ModelFuncs.pde_ModelFuncs_3var import select_model_txy
from ..TrainingSelects.pde_trainingSelect_3var import PINNtrainSelect_txy

""" Class containing ode solution information
    returned on solveODE calls
"""
class pde_txy_solution:
    
    def __init__(self, eqn, t_order, initial_t, setup_boundaries, t_bdry, x_bdry, y_bdry, N_pde, N_iv, 
                 epochs, net_layers, net_units, constraint, model, flag):

        self._eqn = eqn
        self._setup_boundaries = setup_boundaries
        self._initial_t = initial_t
        self._t_bdry = t_bdry
        self._x_bdry = x_bdry
        self._y_bdry = y_bdry
        self._N_pde = N_pde
        self._N_iv = N_iv
        self._epochs = epochs
        self._t_order = t_order
        self._flag = flag
        self._constraint = constraint
        self._net_layers = net_layers
        self._net_units = net_units
        self._model = model

        if self._model == None:
            self._model = select_model_txy(t_bdry, x_bdry, y_bdry, t_order, initial_t, net_layers, net_units, constraint, setup_boundaries)
        
        self._pde_points, self._init_points = defineCollocationPoints_txy(self._t_bdry, self._x_bdry, self._y_bdry,
                                                                                self._initial_t, self._t_order, self._N_pde, self._N_iv)

        self._epoch_loss, self._iv_loss, self._bc_loss, self._pde_loss, self._model = PINNtrainSelect_txy(self._pde_points, 
        self._init_points, self._epochs, self._eqn, self._t_order, self._N_pde,
        self._N_iv, self._setup_boundaries, self._model, self._constraint, self._flag)

        # Grid where to evaluate the model
        self._t_count = math.floor(t_bdry[1]-t_bdry[0])
        l, m, n = self._t_count, 100, 100
        t = np.linspace(self._t_bdry[0], self._t_bdry[1],l)
        t_4D = np.linspace(self._t_bdry[0], self._t_bdry[1],m)
        y = np.linspace(self._y_bdry[0], self._y_bdry[1], m)
        x = np.linspace(self._x_bdry[0], self._x_bdry[1], n)
        self._T, self._X, self._Y = np.meshgrid(t, x, y, indexing='ij')
        self._t, self._x, self._y = np.meshgrid(t_4D, x, y, indexing='ij')

        self._solutionPred = self._model([np.expand_dims(self._T.flatten(), axis=1),
                np.expand_dims(self._X.flatten(), axis=1), np.expand_dims(self._Y.flatten(), axis=1)])
        self._solutionPred = np.reshape(self._solutionPred, (l, m, n))

        self._solutionPred_4D = self._model([np.expand_dims(self._t.flatten(), axis=1),
                np.expand_dims(self._x.flatten(), axis=1), np.expand_dims(self._y.flatten(), axis=1)])
        self._solutionPred_4D = np.reshape(self._solutionPred_4D, (m, m, n))


    def get_equation(self):
        return self._eqn
    
    def get_initial_t(self):
        return self._initial_t
    
    def get_t_bdry(self):
        return self._t_bdry
    
    def get_N_pde(self):
        return self._N_pde
    
    def get_epochs(self):
        return self._epochs
    
    def get_t_order(self):
        return self._t_order
    
    def get_flag(self):
        return self._flag
    
    def get_constraint(self):
        return self._constraint
    
    def get_epoch_loss(self):
        return self._epoch_loss
    
    def get_pde_loss(self):
        return self._pde_loss
    
    def get_iv_loss(self):
        return self._iv_loss
    
    def get_bc_loss(self):
        return self._bc_loss
    
    def get_t_points(self):
        return self._T
    
    def get_x_points(self):
        return self._X
    
    def get_solution_prediction(self):
        return self._solutionPred
    
    def get_model(self):
        return self._model
    
    def get_pde_points(self):
        return self._pde_points
    
    
    def plot_epoch_loss(self, filetitle = "PDE-Epoch-Loss"):
        pde_Plotters.plot_epoch_loss(self._epoch_loss, self._epochs, filetitle)
        return
    
    def plot_iv_loss(self, filetitle = "PDE-IV-Loss"):
        pde_Plotters.plot_iv_loss(self._iv_loss, self._epochs, filetitle)
        return
    
    def plot_bc_loss(self, filetitle = "PDE-BC-Loss"):
        pde_Plotters.plot_bc_loss(self._bc_loss, self._epochs, filetitle)
        return

    def plot_pde_loss(self, filetitle = "PDE-DE-Loss"):
        pde_Plotters.plot_pde_loss(self._pde_loss, self._epochs, filetitle)
        return
    
    def plot_all_losses(self, filetitle = "PDE-All-Losses"):
        pde_Plotters.plot_all_losses(self._epoch_loss, self._pde_loss, self._iv_loss, self._bc_loss, self._epochs, filetitle)
    
    def plot_solution_prediction(self, filetitle = "PDE-solution-pred"):
        pde_Plotters.plot_solution_prediciton_3var(self._t_count, self._X, self._Y, self._solutionPred, "tx", filetitle)
        return
    
    def plot_solution_prediction_test(self, filetitle = "PDE-solution-pred-test"):
        pde_Plotters.plot_solution_prediciton_3var(self._t_count, self._X, self._Y, self._solutionPred_4D, "tx", filetitle)
        return
    
    def plot_solution_prediction_3D(self, filetitle = "PDE-solution-Pred-3D"):
        pde_Plotters.plot_4D(self._t, self._x, self._y, self._solutionPred_4D, "txy", filetitle)
        return