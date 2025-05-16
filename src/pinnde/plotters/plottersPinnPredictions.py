import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_solution_prediction_time1D(model):
    network = model.get_network()
    domain = model.get_domain()
    time_range = domain.get_timeRange()

    T, X = np.meshgrid(np.linspace(time_range[0], time_range[1], 200), 
                       np.linspace(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0], 200), indexing='ij')
    
    sols = network([np.expand_dims(T.flatten(), axis=1), np.expand_dims(X.flatten(), axis=1)])
    sols = np.reshape(sols, (200, 200))

    plt.figure()
    plt.contourf(T, X, sols, 200, cmap=plt.cm.jet)
    plt.title('Neural network solution')
    plt.xlabel('t')
    plt.ylabel('x1')
    plt.colorbar()
    plt.savefig("PDE-solution-pred")
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

def plot_solution_prediction_time2D(model):
    network = model.get_network()
    domain = model.get_domain()
    time_range = domain.get_timeRange()
    interval_dist = (time_range[1] - time_range[0])/3
    intervals = [time_range[0], time_range[0]+interval_dist, time_range[0]+2*interval_dist, time_range[1]]

    X1, X2 = np.meshgrid(np.linspace(domain.get_min_dim_vals()[0], domain.get_max_dim_vals()[0], 200), 
                         np.linspace(domain.get_min_dim_vals()[1], domain.get_max_dim_vals()[1], 200), indexing='ij')
    max = 0
    min = 0
    sols = []
    for i in range(4):
        sol_pts = np.linspace(intervals[i], intervals[i], 200*200)
        sol_pts = sol_pts.reshape((200*200), 1)
        sol_pred = network([sol_pts, np.expand_dims(X1.flatten(), axis=1), np.expand_dims(X2.flatten(), axis=1)])
        sol_pred = np.reshape(sol_pred, (200, 200))
        sols.append(sol_pred)
        if (np.max(sols) > max):
            max = np.max(sols)
        if(np.min(sols) < min):
            min = np.min(sols)

    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        gridspec_kw={'hspace': 0.5, 'wspace': 0.4}  # Vertical/horizontal spacing
    )


    normalizer = Normalize(min, max)
    im = cm.ScalarMappable(norm=normalizer, cmap=plt.cm.jet)


    # Plot data and set subplot titles
    for i, ax in enumerate(axes.flat):
        ax.contourf(X1, X2, sols[i], 200, cmap=plt.cm.jet, norm=normalizer)
        ax.set_title(f"t={intervals[i]:.2f}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    # Add shared colorbar with padding
    fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.08)

    # Adjust layout to leave space for the overall title
    fig.tight_layout(rect=[0, 0, 1, 0.92])  # Reserve 8% space at the top

    # Add overall title
    fig.suptitle('Neural network solution', fontsize=14, y=0.98)
    plt.savefig("PDE-solution-pred")
    plt.clf()
    return