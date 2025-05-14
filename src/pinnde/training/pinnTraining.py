import tensorflow as tf
import numpy as np
import ast

def trainStep(eqns, clps, bcs, network, dim):
    return

def trainStepTime(eqns, clps, bcs, ics, network, dim):

    t_clp = clps[:,0:1]
    clps_group = [t_clp]
    for i in range(dim):
        globals()[f"x{i+1}_clp"] = clps[:,i+1:i+2]
        clps_group.append(globals()[f"x{i+1}_clp"])
    
    t_ics = ics[:,0:1]
    maxi = 0
    ics_group = [t_ics]
    for i in range(dim):
        globals()[f"x{i+1}_ics"] = ics[:,i+1:i+2]
        ics_group.append(globals()[f"x{i+1}_ics"])
        maxi = i+1

    for i in range(maxi, maxi+len(eqns)+1):
        globals()[f"u{i+1}_ics"] = ics[:,i+1:i+2]
        ics_group.append(globals()[f"u{i+1}_ics"])

    t_bcs = bcs[:,0:1]
    u_bcs = bcs[:,-1:]
    bcs_group = [t_bcs]
    for i in range(dim):
        globals()[f"x{i+1}_bc"] = bcs[:,i+1:i+2]
        bcs_group.append(globals()[f"x{i+1}_bc"])
    bcs_group.append(u_bcs)

    # Outer gradient for tuning network parameters
    with tf.GradientTape() as tape:
        # # Inner gradient for derivatives of u wrt x and t
        with tf.GradientTape(persistent=True) as tape1:
            for col in clps_group:
                tape1.watch(col)
            for col in bcs_group:
                tape1.watch(col)
            for col in ics_group:
                tape1.watch(col)
    
            u = network(clps_group)
            ut = tape1.gradient(u, clps_group[0])
            utt = tape1.gradient(ut, clps_group[0])
            uttt = tape1.gradient(utt, clps_group[0])
            first_ders = [ut]
            second_ders = [utt]
            third_ders = [uttt]
            for i in range(dim):
                globals()[f"ux{i+1}"] = tape1.gradient(u, clps_group[i+1])
                first_ders.append(globals()[f"ux{i+1}"])
                globals()[f"ux{i+1}x{i+1}"] = tape1.gradient(globals()[f"ux{i+1}"], clps_group[i+1])
                second_ders.append(globals()[f"ux{i+1}x{i+1}"])
                globals()[f"ux{i+1}x{i+1}x{i+1}"] = tape1.gradient(globals()[f"ux{i+1}x{i+1}"], clps_group[i+1])
                third_ders.append(globals()[f"ux{i+1}x{i+1}x{i+1}"])

        t = clps_group[0]

        CLPloss = 0
        for i in range(len(eqns)):
            parse_tree = ast.parse(eqns[0], mode = "eval")
            eqnparse = eval(compile(parse_tree, "<string>", "eval"))
            CLPloss += tf.reduce_mean(tf.square(eqnparse))
            


    # print(ics)
    # print(ics_group)
    # print(clps)
    # print(clps_group)
    # print(bcs)
    # print(bcs_group)
        loss = CLPloss
    print(loss)
    grads = tape.gradient(loss, network.trainable_variables)
    return CLPloss, 0, 0, grads