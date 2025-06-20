import numpy as np
import matplotlib.pyplot as plt
from ..domain import NRect

def plot_epoch_loss(model):
    plt.figure()
    plt.semilogy(np.linspace(1, model.get_epochs(), model.get_epochs()),model.get_epoch_loss())
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title("Epoch loss")
    if (isinstance(model.get_domain(), NRect) and (model.get_domain().get_dim() == 1)):
        plt.savefig("ODE-epoch-loss")
    else:
        plt.savefig("PDE-epoch-loss")
    plt.clf()
    return