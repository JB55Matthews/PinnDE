import tensorflow as tf
import numpy as np
import ast

def trainStep(eqns, clps, bcs, network, dim, bdry_type):
    return

def trainStepTime(eqns, clps, bcs, ics, network, dim, bdry_type, t_orders):

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

    for i in range(maxi, maxi+t_orders[0]):
        globals()[f"u{i+1}_ics"] = ics[:,i+1:i+2]
        ics_group.append(globals()[f"u{i+1}_ics"])
    

    t_bcs = bcs[:,0:1]
    u_bcs = bcs[:,-1:]
    bcs_group = [t_bcs]
    for i in range(dim):
        globals()[f"x{i+1}_bc"] = bcs[:,i+1:i+2]
        bcs_group.append(globals()[f"x{i+1}_bc"])
    # bcs_group.append(u_bcs)

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
    
            u = tf.cast(network(clps_group), tf.float32)
            ut = tf.cast(tape1.gradient(u, clps_group[0]), tf.float32)
            utt = tf.cast(tape1.gradient(ut, clps_group[0]), tf.float32)
            uttt = tf.cast(tape1.gradient(utt, clps_group[0]), tf.float32)
            first_ders = [ut]
            second_ders = [utt]
            third_ders = [uttt]
            for i in range(dim):
                globals()[f"x{i+1}"] = tf.cast(clps_group[i+1], tf.float32)
                globals()[f"ux{i+1}"] = tf.cast(tape1.gradient(u, clps_group[i+1]), tf.float32)
                # first_ders.append(globals()[f"ux{i+1}"])
                globals()[f"ux{i+1}x{i+1}"] = tf.cast(tape1.gradient(globals()[f"ux{i+1}"], clps_group[i+1]), tf.float32)
                # second_ders.append(globals()[f"ux{i+1}x{i+1}"])
                globals()[f"ux{i+1}x{i+1}x{i+1}"] = tf.cast(tape1.gradient(globals()[f"ux{i+1}x{i+1}"], clps_group[i+1]), tf.float32)
                # third_ders.append(globals()[f"ux{i+1}x{i+1}x{i+1}"])

        t = tf.cast(clps_group[0], tf.float32)

        CLPloss = 0
        parse_tree = ast.parse(eqns[0], mode = "eval")
        eqnparse = eval(compile(parse_tree, "<string>", "eval"))
        CLPloss += tf.reduce_mean(tf.square(eqnparse))
        CLPloss = tf.cast(CLPloss, tf.float32)

        BCloss = 0
        if bdry_type == 1:
            pass

        elif bdry_type == 2:
            # print(bcs)
            # print("break")
            # print(bcs_group)
            u_bc_pred = network(bcs_group)
            u_bcs = tf.cast(u_bcs, tf.float32)
            BCloss = tf.reduce_mean(tf.square(u_bcs-u_bc_pred))

        elif bdry_type == 3:
            pass

        ICloss = 0
        with tf.GradientTape(persistent=True) as tape2:
            for col in ics_group:
                tape2.watch(col)

            u_init_pred = network(ics_group[:dim+1])
            
            ICloss = tf.reduce_mean(tf.square(u_init_pred - tf.cast(ics_group[dim+1], tf.float32)))
        
            for i in range(t_orders[0]-1):
                next_pred = tf.cast(tape2.gradient(u_init_pred, ics_group[0]), tf.float32)
                ICloss += tf.reduce_mean(tf.square(next_pred - tf.cast(ics_group[(dim+1)+(i+1)], tf.float32)))
                u_init_pred = next_pred
        
    
        # print(ics)
        # print(ics_group)
        # print(clps)
        # print(clps_group)
        # print(bcs)
        # print(bcs_group)
        loss = CLPloss + BCloss + ICloss
    
    grads = tape.gradient(loss, network.trainable_variables)
    return CLPloss, BCloss, ICloss, grads