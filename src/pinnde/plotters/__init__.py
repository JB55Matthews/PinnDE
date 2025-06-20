__all__= [
    "plot_solution_prediction_time2D",
    "plot_solution_prediction_time1D",
    "plot_solution_prediction_2D",
    "plot_epoch_loss",
    "plot_solution_prediction_1D",
    "timesteptest"
]

from .plotPinnPredictions import plot_solution_prediction_time2D, \
plot_solution_prediction_2D, plot_solution_prediction_time1D, plot_solution_prediction_1D, timesteptest
from .plotLosses import plot_epoch_loss