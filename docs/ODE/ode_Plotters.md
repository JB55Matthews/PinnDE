# ode_Plotters

Functions for plotting data in ODE solution classes.

While available to user, not meant to be used. Meant to be used through
object returned from solveODE calls, then calling plotting function of object
calls plotting functions with correct data. See examples for how this is done.

We encourage users to plot data by getting data from object (epoch loss, solution prediciton,
etc.), and then making specific plots for each problem. These provide quick, easy to visualize
plots of data from model available from solving function. 


::: src.pinnde.ODE.ode_Plotters
    options:
        members_order: source
    rendering:
      show_root_heading: yes
    selection:
      members:
        - plot_solution_prediction
        - plot_epoch_loss
        - plot_ivp_loss
        - plot_de_loss
        - plot_all_losses
        - plot_predicted_exact
        - plot_solution_prediction_system
        - plot_predicted_exact_system
        