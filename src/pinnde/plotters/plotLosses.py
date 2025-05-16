import numpy as np
import matplotlib.pyplot as plt

def plot_epoch_loss(model):
    plt.figure()
    plt.semilogy(np.linspace(1, model.get_epochs(), model.get_epochs()),model.get_epoch_loss())
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title("Epoch loss")
    plt.savefig("PDE-epoch-loss")
    plt.clf()
    return