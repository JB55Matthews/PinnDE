import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_epoch_loss(epoch_loss, epochs, title):
    """
    Plotting epoch loss of trained model. Saves image in current directory.

    Args:
        epoch_loss (list): Total loss from training
        epochs (int): Epochs trained for
        title (string): File title

    No returns

    """
    plt.figure()
    plt.semilogy(np.linspace(1, epochs, epochs),epoch_loss)
    plt.title("Epoch loss")
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.savefig(title)
    plt.clf()
    return

def plot_iv_loss(iv_loss, epochs, title):
    """
    Plotting just initial value loss of trained model. Will be zero for problems in x and y or hard constraints.
    Saves image in current directory.

    Args:
        iv_loss (list): iv loss from training
        epochs (int): Epochs trained for
        title (string): File title

    No returns
    """
    plt.figure()
    plt.semilogy(np.linspace(1, epochs, epochs),iv_loss)
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('IV loss')

    plt.savefig(title)
    plt.clf()
    return

def plot_bc_loss(bc_loss, epochs, title):
    """
    Plotting just boundary value loss of trained model. Will be zero for hard constraints.
    Saves image in current directory.

    Args:
        bc_loss (list): bc loss from training
        epochs (int): Epochs trained for
        title (string): File title

    No returns
    """
    plt.figure()
    plt.semilogy(np.linspace(1, epochs, epochs),bc_loss)
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('BC loss')

    plt.savefig(title)
    plt.clf()
    return

def plot_pde_loss(de_loss, epochs, title):
    """
    Plotting just differential equation loss of trained model. Saves image in current directory.

    Args:
        de_loss (list): de loss from training
        epochs (int): Epochs trained for
        title (string): File title

    No returns
    """
    plt.figure()
    plt.semilogy(np.linspace(1, epochs, epochs),de_loss)
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('PDE loss')

    plt.savefig(title)
    plt.clf()
    return

def plot_all_losses(epoch_loss, de_loss, ivp_loss, bc_loss, epochs, title):

    plt.figure()
    plt.semilogy(np.linspace(1, epochs, epochs), ivp_loss, label='IV Loss')
    plt.semilogy(np.linspace(1, epochs, epochs), de_loss, label='DE Loss')
    plt.semilogy(np.linspace(1, epochs, epochs), bc_loss, label='BC Loss')
    plt.semilogy(np.linspace(1, epochs, epochs), epoch_loss, label='Total Loss')
    plt.grid()
    plt.xlabel('Epochs')   
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(title)
    plt.clf()
    return

def plot_solution_prediction(T, X, solPred, flag, title):
    """
    Plotting predicted solution of trained model as 2D contour plot. Saves image in current directory.

    Args:
        T (list): Sampled points along T or X meshgrided with X or Y for evaluating model
        X (list): Sampled points along X or Y meshgrided with T or X for evaluating model
        solPred (list): Solution prediction of model
        flag (string): Flag which determines whether to plot axis T/X or X/Y
        title (string): File title

    No returns
    """
    plt.figure()
    plt.contourf(T, X, solPred, 100, cmap=plt.cm.jet)
    plt.title('Neural network solution')
    if flag == "tx":
        plt.xlabel('t')
        plt.ylabel('x')
    elif flag == "xy":
        plt.xlabel("x")
        plt.ylabel("y")
    plt.colorbar()

    plt.savefig(title)
    plt.clf()
    return

def plot_3D(T, X, solPred, flag, title):
    """
    Plotting predicted solution of trained model in 3 dimensions. Saves image in current directory.

    Args:
        T (list): Sampled points along T or X meshgrided with X or Y for evaluating model
        X (list): Sampled points along X or Y meshgrided with T or X for evaluating model
        solPred (list): Solution prediction of model
        flag (string): Flag which determines whether to plot axis T/X or X/Y
        title (string): File title

    No returns
    """
    
    if flag == "tx":
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot_surface(T, X, solPred, cmap = plt.cm.coolwarm)
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_zlabel("u")
        ax.set_title('Neural network solution of U(t,x)')
        plt.savefig(title)
        plt.clf()

    elif flag == "xy":
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot_surface(T, X, solPred, cmap = plt.cm.coolwarm)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u")
        ax.set_title('Neural network solution of U(x,y)')
        plt.savefig(title)
        plt.clf()
    return

def plot_predicted_exact(T, X, solPred, exact, flag, title):
    """
    Plotting predicted solution of model vs input function of solution. Saves image in current directory.

    Args:
        T (list): Sampled points along T or X meshgrided with X or Y for evaluating model
        X (list): Sampled points along X or Y meshgrided with T or X for evaluating model
        solPred (list): Solution prediction of model
        exact (list): Values of exact equation evaluated at same points
        flag (string): Flag which determines whether to plot axis T/X or X/Y
        title (string): File title

    No returns
    """

    plt.figure(figsize=(14, 5))

    plt.subplot(131)
    plt.contourf(T, X, solPred, 100, cmap=plt.cm.jet)
    plt.title('Neural network solution')
    if flag == "tx":
        plt.xlabel('t')
        plt.ylabel('x')
    elif flag == "xy":
        plt.xlabel("x")
        plt.ylabel("y")
    plt.colorbar()

    plt.subplot(132)
    plt.contourf(T, X, exact, 100, cmap=plt.cm.jet)
    plt.title('Exact solution')
    if flag == "tx":
        plt.xlabel('t')
        plt.ylabel('x')
    elif flag == "xy":
        plt.xlabel("x")
        plt.ylabel("y")
    plt.colorbar()

    plt.subplot(133)
    plt.contourf(T, X, solPred-exact, 100, cmap=plt.cm.jet)
    plt.title(f'MSE: {tf.reduce_mean((solPred-exact)**2): 4.4g}')
    if flag == "tx":
        plt.xlabel('t')
        plt.ylabel('x')
    elif flag == "xy":
        plt.xlabel("x")
        plt.ylabel("y")
    plt.colorbar()

    plt.savefig(title)
    plt.clf()
    return
    

def plot_solution_prediciton_3var(t_count, X, Y, solPred, flag, title):
    #print("Starting")
    # rows = (t_count//2)+1
    # for i in range(0,t_count):
    #     plt.subplot(1,3,1)
    #     plt.contourf(X[0,], Y[0,], solPred[0,], 100, cmap=plt.cm.jet)
    #     plt.title("Neural network solution, T={}".format(0))
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     #plt.colorbar()
    #     plt.savefig(title)
    #     plt.clf()
    fig, axes = plt.subplots(nrows=((t_count//2)+1), ncols=2, figsize=(12, 8))

    i = 0
    for row in axes:
        for ax in row:
            if i >= t_count:
                break
            ax.contourf(X[i,], Y[i,], solPred[i,], 100, cmap=plt.cm.jet)
            # print(str(i) + ":")
            # print(solPred[i,])
            ax.set_title("Neural network solution, T={}".format(i))
            i += 1
    plt.tight_layout()
    plt.savefig(title)
    plt.clf()
    return

def plot_4D(T, X, Y, solPred, flag, title):
    # print("T")
    # print(T)
    # print()
    # print("X")
    # print(X)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    p = ax.scatter(T, X, Y, c=solPred, cmap = plt.cm.coolwarm)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    ax.set_title('Neural network solution of U(t,x,y)')
    fig.colorbar(p)
    plt.savefig(title)
    plt.clf()