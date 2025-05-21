import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_solution_prediction_time1D(model):
    eqns = model.get_eqns()
    network = model.get_network()
    domain = model.get_domain()
    time_range = domain.get_timeRange()

    T, X = np.meshgrid(np.linspace(time_range[0], time_range[1], 200), 
                       np.linspace(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0], 200), indexing='ij')
    
    sols = network([np.expand_dims(T.flatten(), axis=1), np.expand_dims(X.flatten(), axis=1)])

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
    network = model.get_network()
    domain = model.get_domain()
    
    X1, X2 = np.meshgrid(np.linspace(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0], 200), 
                         np.linspace(domain.get_min_dim_vals()[1], domain.get_max_dim_vals()[1], 200), indexing='ij')
    
    sols = network([np.expand_dims(X1.flatten(), axis=1), np.expand_dims(X2.flatten(), axis=1)])
    sols = np.reshape(sols, (200, 200))

    plt.figure()
    plt.contourf(X1, X2, sols, 200, cmap=plt.cm.jet)
    plt.title('Neural network solution')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()
    plt.savefig("PDE-solution-pred")
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

    max_val = -np.inf
    min_val = np.inf
    sols = []
    
    for i in range(4):
        # Get predictions
        t = np.full_like(X1.flatten(), intervals[i])
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
    return