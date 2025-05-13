import tensorflow as tf
import numpy as np


def trainStep(eqns, clps, bcs, network, dim):
    return

def trainStepTime(eqns, clps, bcs, ics, network, dim):
    clps_group = []
    inits_group = []
    print(ics)
    t_clp = clps[:,0:1]
    clps_group = [t_clp]
    for i in range(dim):
        globals()[f"x{i+1}_clp"] = clps[:,i+1:i+2]
    clps_group.append(globals()[f"x{i+1}_clp"])
    return