import ast
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ..models.pinn import pinn
from ..models.deeponet import deeponet
from ..models.invpinn import invpinn
import tensorflow as tf

def plot_solution_prediction_1D(model):
    """
    Plots the predicted solution for an 1 spatial dimension (ODE) trained model.

    Args:
        model (model): Model which has been trained on a 1 dimensional NRect.
    """
    eqns = model.get_eqns()
    network = model.get_network()
    domain = model.get_domain()

    t = np.linspace(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0], 200)
    
    if isinstance(model, pinn):
        sols = network(np.expand_dims(t, axis=1))
    elif isinstance(model, deeponet):
        x = np.linspace(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0], model.get_data().get_n_sensors())
        amplitudes = np.random.randn(3, 1)
        phases = -np.pi*np.random.rand(3, 1) + np.pi/2
        u = 0.0*x
        for i in range(3):
            u += amplitudes[i]*tf.sin((i+1)*np.expand_dims(x, axis=0)+ phases[i])
        sensors = u.numpy()    
        sols = network([np.expand_dims(t, axis=1), sensors])
        sols = tf.squeeze(sols, axis=1)
    
    elif (isinstance(model, invpinn)):
        sols = network(np.expand_dims(t, axis=1))[:-len(model.get_constants())]

    if len(eqns) == 1:
        plt.figure()
        plt.plot(t, sols)
        plt.title('Neural network solution')
        plt.grid()
        plt.xlabel('x1')
        plt.ylabel('u')
        plt.savefig("ODE-solution-pred")
        plt.clf()

    elif len(eqns) > 1:
        plt.figure()
        for e in range(len(eqns)):
            globals()[f"sols{e+1}"] = sols[e]
            plt.plot(t, globals()[f"sols{e+1}"])
        plt.title(f'Neural network solution')
        plt.grid()
        plt.xlabel('x1')
        plt.ylabel('u')
        plt.savefig(f"ODE-solution-pred")
        plt.clf()
    return

def plot_solution_prediction_time1D(model):
    """
    Plots the predicted solution for an 1+1 PDE trained model.

    Args:
        model (model): Model which has been trained for a 1+1 equation.
    """
    eqns = model.get_eqns()
    network = model.get_network()
    domain = model.get_domain()
    time_range = domain.get_timeRange()

    T, X = np.meshgrid(np.linspace(time_range[0], time_range[1], 200), 
                       np.linspace(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0], 200), indexing='ij')
    
    if (isinstance(model, pinn)):
        sols = network([np.expand_dims(T.flatten(), axis=1), np.expand_dims(X.flatten(), axis=1)])

    elif (isinstance(model, deeponet)):
        x = np.linspace(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0], model.get_data().get_n_sensors())
        amplitudes = np.random.randn(3, 1)
        phases = -np.pi*np.random.rand(3, 1) + np.pi/2
        u = 0.0*x
        for i in range(3):
            u += amplitudes[i]*tf.sin((i+1)*np.expand_dims(x, axis=0)+ phases[i])
        sensors = u.numpy()
        sols = network([np.expand_dims(T.flatten(), axis=1), np.expand_dims(X.flatten(), axis=1), sensors])

    elif (isinstance(model, invpinn)):
        sols = network([np.expand_dims(T.flatten(), axis=1), np.expand_dims(X.flatten(), axis=1)])[:-len(model.get_constants())]

    if len(eqns) == 1:
        sols = np.reshape(sols, (200, 200))
        plt.figure()
        plt.contourf(T, X, sols, 200, cmap=plt.cm.jet)
        plt.title('Neural network solution')
        plt.xlabel('t')
        plt.ylabel('x1')
        plt.colorbar()
        plt.savefig("PDE-solution-pred")
        plt.clf()

    elif len(eqns) > 1:
        for e in range(len(eqns)):
            globals()[f"sols{e+1}"] = np.reshape(sols[e], (200, 200))
            plt.figure()
            plt.contourf(T, X, globals()[f"sols{e+1}"], 200, cmap=plt.cm.jet)
            plt.title(f'Neural network solution - u{e+1}')
            plt.xlabel('t')
            plt.ylabel('x1')
            plt.colorbar()
            plt.savefig(f"PDE-solution-pred-u{e+1}")
            plt.clf()
    return

def plot_solution_prediction_2D(model):
    """
    Plots the predicted solution for an 2 spatial dimension PDE trained model.

    Args:
        model (model): Model which has been trained for a 2D equation.
    """
    network = model.get_network()
    domain = model.get_domain()
    eqns = model.get_eqns()
    
    X1, X2 = np.meshgrid(np.linspace(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0], 200), 
                         np.linspace(domain.get_min_dim_vals()[1], domain.get_max_dim_vals()[1], 200), indexing='ij')
    

    if isinstance(model, pinn):
        sols = network([np.expand_dims(X1.flatten(), axis=1), np.expand_dims(X2.flatten(), axis=1)])
    elif isinstance(model, deeponet):
        x = np.linspace(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0], model.get_data().get_n_sensors())
        y = np.linspace(domain.get_min_dim_vals()[1], domain.get_max_dim_vals()[1], model.get_data().get_n_sensors())
        amplitudes = np.random.randn(3, 1)
        phases = -np.pi*np.random.rand(3, 1) + np.pi/2
        u = 0.0*x
        for i in range(3):
            u += amplitudes[i]*tf.sin((i+1)*np.expand_dims(x, axis=0)+ phases[i])*tf.sin((i+1)*np.expand_dims(y, axis=0)+ phases[i])
        sensors = u.numpy()
        sols = network([np.expand_dims(X1.flatten(), axis=1), np.expand_dims(X2.flatten(), axis=1), sensors])

    elif (isinstance(model, invpinn)):
        sols = network([np.expand_dims(X1.flatten(), axis=1), np.expand_dims(X2.flatten(), axis=1)])[:-len(model.get_constants())]

    if len(eqns) == 1:
        sols = np.reshape(sols, (200, 200))

        plt.figure()
        plt.contourf(X1, X2, sols, 200, cmap=plt.cm.jet)
        plt.title('Neural network solution')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.colorbar()
        plt.savefig("PDE-solution-pred")
        plt.clf()
    
    elif len(eqns) > 1:
        for e in range(len(eqns)):
            globals()[f"sols{e+1}"] = np.reshape(sols[e], (200, 200))
            plt.figure()
            plt.contourf(X1, X2, globals()[f"sols{e+1}"], 200, cmap=plt.cm.jet)
            plt.title(f'Neural network solution - u{e+1}')
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.colorbar()
            plt.savefig(f"PDE-solution-pred-u{e+1}")
            plt.clf()
    return

# def plot_solution_prediction_time2D(model):
#     network = model.get_network()
#     domain = model.get_domain()
#     time_range = domain.get_timeRange()
#     interval_dist = (time_range[1] - time_range[0])/3
#     intervals = [time_range[0], time_range[0]+interval_dist, time_range[0]+2*interval_dist, time_range[1]]

#     m = 200

#     x1 = np.linspace(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0], m)
#     x2 = np.linspace(domain.get_min_dim_vals()[1], domain.get_max_dim_vals()[1], m)
 
#     X1, X2 = np.meshgrid(x1, x2, indexing='ij')
#     n = np.shape(x1)[0]

#     points = np.stack([X1.flatten(), X2.flatten()], axis=1)
#     mask = np.array([domain.isInside(pt) for pt in points])
#     mask = mask.reshape(X1.shape)

#     max_val = -np.inf
#     min_val = np.inf
#     sols = []

#     for i in range(4):
#         sol_pts = np.linspace(intervals[i], intervals[i], n*n)
#         sol_pts = sol_pts.reshape((n*n), 1)
#         sol_pred = network([sol_pts, np.expand_dims(X1.flatten(), axis=1), np.expand_dims(X2.flatten(), axis=1)])
#         sol_pred = np.reshape(sol_pred, (n, n))
#         sols.append(sol_pred)
#         if (np.max(sols) > max):
#             max = np.max(sols)
#         if(np.min(sols) < min):
#             min = np.min(sols)

#     fig, axes = plt.subplots(
#         nrows=2, ncols=2,
#         gridspec_kw={'hspace': 0.5, 'wspace': 0.4}  # Vertical/horizontal spacing
#     )


#     normalizer = Normalize(min, max)
#     im = cm.ScalarMappable(norm=normalizer, cmap=plt.cm.jet)

#     # Plot data and set subplot titles
#     for i, ax in enumerate(axes.flat):
#         ax.contourf(X1, X2, sols[i], 200, cmap=plt.cm.jet, norm=normalizer)
#         ax.set_aspect('equal')
#         ax.set_title(f"t={intervals[i]:.2f}")
#         ax.set_xlabel("x1")
#         ax.set_ylabel("x2")
#         ax.set_xlim(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0])
#         ax.set_ylim(domain.get_min_dim_vals()[1], domain.get_max_dim_vals()[1])


#     # Add shared colorbar with padding
#     fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.08)

#     # Adjust layout to leave space for the overall title
#     fig.tight_layout(rect=[0, 0, 1, 0.92])  # Reserve 8% space at the top

#     # Add overall title
#     fig.suptitle('Neural network solution', fontsize=14, y=0.98)
#     plt.savefig("PDE-solution-pred")
#     plt.clf()
#     return

def plot_solution_prediction_time2D(model):
    """
    Plots the predicted solution for an 1+2 PDE trained model.

    Args:
        model (model): Model which has been trained for a 1+2 equation.
    """
    eqns = model.get_eqns()
    network = model.get_network()
    domain = model.get_domain()
    time_range = domain.get_timeRange()
    interval_dist = (time_range[1] - time_range[0])/3
    intervals = [time_range[0], time_range[0]+interval_dist, 
                time_range[0]+2*interval_dist, time_range[1]]


    m = 200  # Grid resolution
    
    # Generate regular grid spanning domain bounds
    x1 = np.linspace(domain.get_min_dim_vals()[0], 
                    domain.get_max_dim_vals()[0], m)
    x2 = np.linspace(domain.get_min_dim_vals()[1], 
                    domain.get_max_dim_vals()[1], m)
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')
    
    # Create mask for elliptical domain
    points = np.stack([X1.flatten(), X2.flatten()], axis=1)
    mask = np.array([domain.isInside(pt) for pt in points])
    mask = mask.reshape(X1.shape)

    if len(eqns) == 1:
        max_val = -np.inf
        min_val = np.inf
        sols = []
        
        for i in range(4):
            # Get predictions
            t = np.full_like(X1.flatten(), intervals[i])
            if isinstance(model, pinn):
                sol_pred = network([t[:, None], 
                                X1.flatten()[:, None], 
                                X2.flatten()[:, None]])
                sol_pred = sol_pred.numpy().reshape(X1.shape)
                
                # Apply domain mask
                sol_pred[~mask] = np.nan  # Mask outside domain
                
                sols.append(sol_pred)
                current_max = np.nanmax(sol_pred)
                current_min = np.nanmin(sol_pred)
                if current_max > max_val:
                    max_val = current_max
                if current_min < min_val:
                    min_val = current_min

            elif isinstance(model, deeponet):
                x = np.linspace(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0], model.get_data().get_n_sensors())
                y = np.linspace(domain.get_min_dim_vals()[1], domain.get_max_dim_vals()[1], model.get_data().get_n_sensors())
                amplitudes = np.random.randn(3, 1)
                phases = -np.pi*np.random.rand(3, 1) + np.pi/2
                u = 0.0*x
                for i in range(3):
                    u += amplitudes[i]*tf.sin((i+1)*np.expand_dims(x, axis=0)+ phases[i])*tf.sin((i+1)*np.expand_dims(y, axis=0)+ phases[i])
                sensors = u.numpy()
                sol_pred = network([t[:, None], 
                                X1.flatten()[:, None], 
                                X2.flatten()[:, None],
                                sensors])
                sol_pred = sol_pred.numpy().reshape(X1.shape)
                
                # Apply domain mask
                sol_pred[~mask] = np.nan  # Mask outside domain
                
                sols.append(sol_pred)
                current_max = np.nanmax(sol_pred)
                current_min = np.nanmin(sol_pred)
                if current_max > max_val:
                    max_val = current_max
                if current_min < min_val:
                    min_val = current_min

            elif isinstance(model, invpinn):
                sol_pred = network([t[:, None], 
                                X1.flatten()[:, None], 
                                X2.flatten()[:, None]])[0]
                sol_pred = sol_pred.numpy().reshape(X1.shape)
                
                # Apply domain mask
                sol_pred[~mask] = np.nan  # Mask outside domain
                
                sols.append(sol_pred)
                current_max = np.nanmax(sol_pred)
                current_min = np.nanmin(sol_pred)
                if current_max > max_val:
                    max_val = current_max
                if current_min < min_val:
                    min_val = current_min

        # Plotting setup
        fig, axes = plt.subplots(2, 2, gridspec_kw={'hspace': 0.5, 'wspace': 0.4})
        normalizer = Normalize(min_val, max_val)
        im = cm.ScalarMappable(norm=normalizer, cmap=plt.cm.jet)

        # Plot each time step
        for i, ax in enumerate(axes.flat):
            contour = ax.contourf(X1, X2, sols[i], 200, 
                                cmap=plt.cm.jet, norm=normalizer)
            ax.set_title(f"t={intervals[i]:.2f}")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_aspect('equal')
            ax.set_xlim(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0])
            ax.set_ylim(domain.get_min_dim_vals()[1], domain.get_max_dim_vals()[1])

        # Add colorbar and titles
        fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.08)
        fig.suptitle('Neural network solution', fontsize=14, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        
        plt.savefig("PDE-solution-pred")
        plt.clf()

    elif len(eqns) > 1:
        for e in range(len(eqns)):
            max_val = -np.inf
            min_val = np.inf
            sols = []
            
            for i in range(4):
                # Get predictions
                t = np.full_like(X1.flatten(), intervals[i])
                sol_pred = network([t[:, None], 
                                X1.flatten()[:, None], 
                                X2.flatten()[:, None]])[e]
                sol_pred = sol_pred.numpy().reshape(X1.shape)
                
                # Apply domain mask
                sol_pred[~mask] = np.nan  # Mask outside domain
                
                sols.append(sol_pred)
                current_max = np.nanmax(sol_pred)
                current_min = np.nanmin(sol_pred)
                if current_max > max_val:
                    max_val = current_max
                if current_min < min_val:
                    min_val = current_min

            # Plotting setup
            fig, axes = plt.subplots(2, 2, gridspec_kw={'hspace': 0.5, 'wspace': 0.4})
            normalizer = Normalize(min_val, max_val)
            im = cm.ScalarMappable(norm=normalizer, cmap=plt.cm.jet)

            # Plot each time step
            for i, ax in enumerate(axes.flat):
                contour = ax.contourf(X1, X2, sols[i], 200, 
                                    cmap=plt.cm.jet, norm=normalizer)
                ax.set_title(f"t={intervals[i]:.2f}")
                ax.set_xlabel("x1")
                ax.set_ylabel("x2")
                ax.set_aspect('equal')
                ax.set_xlim(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0])
                ax.set_ylim(domain.get_min_dim_vals()[1], domain.get_max_dim_vals()[1])

            # Add colorbar and titles
            fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.08)
            fig.suptitle(f'Neural network solution - u{e+1}', fontsize=14, y=0.98)
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            
            plt.savefig(f"PDE-solution-pred-u{e+1}")
            plt.clf()
    return


# def timesteptest(model, steps):
#     eqns = model.get_eqns()
#     network = model.get_network()
#     domain = model.get_domain()
#     time_range = domain.get_timeRange()

#     T, X = np.meshgrid(np.linspace(time_range[0], time_range[1], 200), 
#                        np.linspace(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0], model.get_data().get_n_sensors()), indexing='ij')
#     Tall, Xall = np.meshgrid(np.linspace(time_range[0], time_range[1]*steps, steps*199), 
#                        np.linspace(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0], model.get_data().get_n_sensors()), indexing='ij')
    
#     if (isinstance(model, pinn)):
#         sols = network([np.expand_dims(T.flatten(), axis=1), np.expand_dims(X.flatten(), axis=1)])
#     elif (isinstance(model, deeponet)):
#         x = np.linspace(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0], model.get_data().get_n_sensors())
#         amplitudes = np.random.randn(3, 1)
#         phases = -np.pi*np.random.rand(3, 1) + np.pi/2
#         u = 0.0*x
#         for i in range(3):
#             u += amplitudes[i]*tf.sin((i+1)*np.expand_dims(x, axis=0)+ phases[i])
#         usensor = u.numpy()
#         u = []
#         for i in range(steps):
#             u_c = network([np.expand_dims(T.flatten(), axis=1),
#                         np.expand_dims(X.flatten(), axis=1),
#                         usensor])[:,0]
#             u_c = np.reshape(u_c, (200, model.get_data().get_n_sensors()))

#             # Reconstruct solution
#             uinit = np.repeat(usensor, repeats=200, axis=0)

#             # For form u(t,x) = u0(x) + t/tf*v(t,x)
#             # u_c = uinit + T/time_range[1]*u_c
#             u_c = uinit

#             # # For form u(t,x) = u0(x)*(1-t/tf) + t/tf*v(t,x)
#             # u_c = uinit*(1-T/tfinal) + T/tfinal*u_c
#             u.append(u_c[:-1,])

#             # Next initial condition
#             usensor = u_c[-1:,:]

#         u = np.concatenate(u)


#     if len(eqns) == 1:
#         # sols = np.reshape(sols, (200, 200))
#         plt.figure()
#         plt.contourf(Tall, Xall, u, 200, cmap=plt.cm.jet)
#         plt.title('Neural network solution')
#         plt.xlabel('t')
#         plt.ylabel('x1')
#         plt.colorbar()
#         plt.savefig("timesteptest")
#         plt.clf()

#     elif len(eqns) > 1:
#         for e in range(len(eqns)):
#             globals()[f"sols{e+1}"] = np.reshape(sols[e], (200, 200))
#             plt.figure()
#             plt.contourf(T, X, globals()[f"sols{e+1}"], 200, cmap=plt.cm.jet)
#             plt.title(f'Neural network solution - u{e+1}')
#             plt.xlabel('t')
#             plt.ylabel('x1')
#             plt.colorbar()
#             plt.savefig(f"PDE-solution-pred-u{e+1}")
#             plt.clf()
#     return

# def paper(model, exact_eqn):
#     eqns = model.get_eqns()
#     network = model.get_network()
#     domain = model.get_domain()
#     time_range = domain.get_timeRange()
#     interval_dist = (time_range[1] - time_range[0])/3
#     intervals = [time_range[0], time_range[0]+interval_dist, 
#                 time_range[0]+2*interval_dist, time_range[1]]


#     m = 200  # Grid resolution
    
#     # Generate regular grid spanning domain bounds
#     x1 = np.linspace(domain.get_min_dim_vals()[0], 
#                     domain.get_max_dim_vals()[0], m)
#     x2 = np.linspace(domain.get_min_dim_vals()[1], 
#                     domain.get_max_dim_vals()[1], m)
#     X1, X2 = np.meshgrid(x1, x2, indexing='ij')
    
#     # Create mask for elliptical domain
#     points = np.stack([X1.flatten(), X2.flatten()], axis=1)
#     mask = np.array([domain.isInside(pt) for pt in points])
#     mask = mask.reshape(X1.shape)

#     if len(eqns) == 1:
#         max_val = -np.inf
#         min_val = np.inf
#         sols = []
#         exact_sols = []
#         diff = []
#         difference=0
#         for i in range(4):
#             # Get predictions
#             t = np.full_like(X1.flatten(), intervals[i])
#             if isinstance(model, pinn):
#                 sol_pred = network([t[:, None], 
#                                 X1.flatten()[:, None], 
#                                 X2.flatten()[:, None]])
#                 sol_pred = sol_pred.numpy().reshape(X1.shape)
#                 t_ex = t[:, None]
#                 x1_ex = X1.flatten()[:, None] 
#                 x2_ex = X2.flatten()[:, None]
#                 parse_tree = ast.parse(exact_eqn, mode="eval")
#                 exact_eqn_data = eval(compile(parse_tree, "<string>", "eval"))
#                 exact_eqn_data = exact_eqn_data.reshape(X1.shape)
#                 # exact_sols.append(exact_eqn_data)
#                 # diff.append(sol_pred-exact_eqn_data)
#                 difference += sol_pred - exact_eqn_data
#                 # Apply domain mask
#                 sol_pred[~mask] = np.nan  # Mask outside domain
#                 exact_eqn_data[~mask] = np.nan
#                 diffto = sol_pred - exact_eqn_data
#                 diffto[~mask] = np.nan

#                 # difference += diffto

#                 sols.append(sol_pred)
#                 exact_sols.append(exact_eqn_data)
#                 diff.append(diffto)

#                 current_max = np.nanmax(sol_pred)
#                 current_min = np.nanmin(sol_pred)
#                 if current_max > max_val:
#                     max_val = current_max
#                 if current_min < min_val:
#                     min_val = current_min

      
#         mse = tf.reduce_mean((difference)**2)
#         fig, axes = plt.subplots(2, 2, gridspec_kw={'hspace': 0.5, 'wspace': 0.4})
#         normalizer = Normalize(min_val, max_val)
#         im = cm.ScalarMappable(norm=normalizer, cmap=plt.cm.jet)

#         # Plot each time step
#         for i, ax in enumerate(axes.flat):
#             contour = ax.contourf(X1, X2, sols[i], 200, 
#                                 cmap=plt.cm.jet, norm=normalizer)
#             ax.set_title(f"t={intervals[i]:.2f}")
#             ax.set_xlabel("x1")
#             ax.set_ylabel("x2")
#             ax.set_aspect('equal')
#             ax.set_xlim(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0])
#             ax.set_ylim(domain.get_min_dim_vals()[1], domain.get_max_dim_vals()[1])

#         # Add colorbar and titles
#         fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.08)
#         fig.suptitle('Neural network solution', fontsize=14, y=0.98)
#         fig.tight_layout(rect=[0, 0, 1, 0.92])
        
#         plt.savefig("predpaper")
#         plt.clf()

#         fig, axes = plt.subplots(2, 2, gridspec_kw={'hspace': 0.5, 'wspace': 0.4})
#         normalizer = Normalize(0, 1)
#         im = cm.ScalarMappable(norm=normalizer, cmap=plt.cm.jet)

#         # Plot each time step
#         for i, ax in enumerate(axes.flat):
#             contour = ax.contourf(X1, X2, exact_sols[i], 200, 
#                                 cmap=plt.cm.jet, norm=normalizer)
#             ax.set_title(f"t={intervals[i]:.2f}")
#             ax.set_xlabel("x1")
#             ax.set_ylabel("x2")
#             ax.set_aspect('equal')
#             ax.set_xlim(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0])
#             ax.set_ylim(domain.get_min_dim_vals()[1], domain.get_max_dim_vals()[1])

#         # Add colorbar and titles
#         fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.08)
#         fig.suptitle('Exact solution', fontsize=14, y=0.98)
#         fig.tight_layout(rect=[0, 0, 1, 0.92])
        
#         plt.savefig("exactpaper")
#         plt.clf()

#         fig, axes = plt.subplots(2, 2, gridspec_kw={'hspace': 0.5, 'wspace': 0.4})
#         # normalizer = Normalize(-1, 1)
#         im = cm.ScalarMappable(cmap=plt.cm.jet)

#         # Plot each time step
#         for i, ax in enumerate(axes.flat):
#             contour = ax.contourf(X1, X2, diff[i], 200, 
#                                 cmap=plt.cm.jet)
#             ax.set_title(f"t={intervals[i]:.2f}")
#             ax.set_xlabel("x1")
#             ax.set_ylabel("x2")
#             ax.set_aspect('equal')
#             ax.set_xlim(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0])
#             ax.set_ylim(domain.get_min_dim_vals()[1], domain.get_max_dim_vals()[1])


#         # Add colorbar and titles
#         fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.08)
#         fig.suptitle(f'MSE: {mse: 4.4g}', fontsize=14, y=0.98)
#         fig.tight_layout(rect=[0, 0, 1, 0.92])
        
#         plt.savefig("msepaper")
#         plt.clf()