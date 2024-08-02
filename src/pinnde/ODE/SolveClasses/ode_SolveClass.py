import numpy as np
import ast

from ..ode_trainingSelect import PINNtrainSelect_Standard
from .. import ode_Points
from .. import ode_Plotters


class ode_solution:
    """
    Name of class returned
    """
    
    def __init__(self, eqn, inits, t_bdry, N_pde, epochs, order, net_layers, net_units, constraint, model, flag):
        """
        Constructer for class.

        Args:
            eqn (string): Equation to solve in form of string. function and derivatives represented as "u", "ut", "utt", 
                etc. for including function, i.e. cos(u), use tf.cos(u), or for ln(t), np.log(t). Write equation as would be written in code.
            inits (list): inital data for each deriviatve.
            t_bdry (list): list of two elements, the interval of t to be solved on.
            N_pde (int): Number of randomly sampled collocation points along t which PINN uses in training.
            epochs (int): Number of epochs PINN gets trained for.
            order (int): order of equation
            net_layers (int): Number of internal layers of PINN
            net_units (int): Number of units in each internal layer
            constraint (string): Determines hard constrainting inital conditions or network learning inital conditions. "soft" or "hard"
            model (PINN): User may pass in user constructed network, however no guarentee of correct training.
            flag (string): Flag for internal use to identify which equation is being solved.

        Constructer also generates, t: randomly sampled points along t, and de_points: 
        randomly sampled points for model to train with. Then calls PINNtrainSelect_Standard function to train. Then defines 
        solution prediction from returned model

        Class Functions
        --------------
        Fucntions the user should call to access information from solveODE call. Getter functions and Plotter functions

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
        self._constraint = constraint
        self._net_layers = net_layers
        self._net_units = net_units
        self._model = model

        self._t = np.linspace(self._t_bdry[0], self._t_bdry[1], self._N_pde)

        self._de_points = ode_Points.defineCollocationPoints(self._t_bdry, self._N_pde)

        self._epoch_loss, self._vp_loss, self._de_loss, self._model = PINNtrainSelect_Standard(self._de_points, 
               self._inits, self._t_bdry, self._epochs, self._eqn, self._order, self._net_layers, 
               self._net_units, self._constraint, self._model, self._flag)

        self._solutionPred = self._model(np.expand_dims(self._t, axis=1))[:,0]

    def get_equation(self):
        """
        Return input equation
        """
        return self._eqn
    
    def get_inits(self):
        """
        Return input init_data
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
        Return internal flag used
        """
        return self._flag
    
    def get_constraint(self):
        """
        Return input constraint
        """
        return self._constraint
    
    def get_epoch_loss(self):
        """
        Return model epoch loss
        """
        return self._epoch_loss
    
    def get_de_loss(self):
        """
        Return model de loss
        """
        return self._de_loss
    
    def get_vp_loss(self):
        """
        Return model ivp/bvp loss
        """
        return self._vp_loss
    
    def get_t_points(self):
        """
        Return sampled t points
        """
        return self._t
    
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
    
    def get_de_points(self):
        """
        Return sampled de points
        """
        return self._de_points
    
    def get_net_units(self):
        """
        Return input net units
        """
        return self._net_units
    
    def get_net_layers(self):
        """
        Return input net layers

        Plotters
        ---------
        """
        return self._net_layers
    
    
    def plot_epoch_loss(self, filetitle = "ODE-Epoch-Loss"):
        """
        Calls ode_Plotters.plot_epoch_loss with correct data

        Args:
            filetitle (string): Title of saved file
        """
        ode_Plotters.plot_epoch_loss(self._epoch_loss, self._epochs, filetitle)
        return
    
    def plot_vp_loss(self, filetitle = "ODE-IVP-Loss"):
        """
        Calls ode_Plotters.plot_ivp_loss with correct data

        Args:
            filetitle (string): Title of saved file
        """
        ode_Plotters.plot_ivp_loss(self._vp_loss, self._epochs, filetitle)
        return

    def plot_de_loss(self, filetitle = "ODE-DE-Loss"):
        """
        Calls ode_Plotters.plot_de_loss with correct data

        Args:
            filetitle (string): Title of saved file
        """
        ode_Plotters.plot_de_loss(self._de_loss, self._epochs, filetitle)
        return
    
    def plot_all_losses(self, filetitle = "ODE-All-Losses"):
        """
        Calls ode_Plotters.plot_all_losses with correct data

        Args:
            filetitle (string): Title of saved file
        """
        ode_Plotters.plot_all_losses(self._epoch_loss, self._de_loss, self._vp_loss, self._epochs, filetitle)
    
    def plot_solution_prediction(self, filetitle = "ODE-solution-pred"):
        """
        Calls ode_Plotters.plot_solution_prediction with correct data

        Args:
            filetitle (string): Title of saved file
        """
        ode_Plotters.plot_solution_prediction(self._t, self._solutionPred, filetitle)
        return
    
    def plot_predicted_exact(self, exact_eqn, filetitle = "ODE-SolPred-Exact"):
        """
        Calls ode_Plotters.plot_predicted_exact with correct data

        Args:
            exact_eqn (lambda): Exact solution of equation as a python lambda function
            filetitle (string): Title of saved file
        """
        exact_output = []
        for i in self._t:
            t = i
            parse_tree = ast.parse(exact_eqn, mode="eval")
            eqn = eval(compile(parse_tree, "<string>", "eval"))
            exact_output.append(eqn)
        ode_Plotters.plot_predicted_exact(self._t, self._solutionPred, exact_output, filetitle)
        return