import tensorflow as tf
import numpy as np
import ast

def trainStep(eqns, clps, bcs, network, boundary):
    dim = boundary.get_domain().get_dim()
    bdry_type = boundary.get_bdry_type()
    bdry_components = boundary.get_domain().get_bdry_components()
    bdry_component_sizes = boundary.get_domain().get_bdry_component_size()

    clps_group = []
    for i in range(dim):
        globals()[f"x{i+1}_clp"] = clps[:,i:i+1]
        clps_group.append(globals()[f"x{i+1}_clp"])  

    u_bcs = bcs[:,-1:]
    bcs_group = []
    for i in range(dim):
        globals()[f"x{i+1}_bc"] = bcs[:,i:i+1]
        bcs_group.append(globals()[f"x{i+1}_bc"])

    # Outer gradient for tuning network parameters
    with tf.GradientTape() as tape:
        # # Inner gradient for derivatives of u wrt x and t
        with tf.GradientTape(persistent=True) as tape1:
            for col in clps_group:
                tape1.watch(col)
            for col in bcs_group:
                tape1.watch(col)

            if len(eqns) == 1:
                u = tf.cast(network(clps_group), tf.float32)
                for i in range(dim):
                    globals()[f"x{i+1}"] = tf.cast(clps_group[i], tf.float32)
                    globals()[f"ux{i+1}"] = tf.cast(tape1.gradient(u, clps_group[i]), tf.float32)
                    globals()[f"ux{i+1}x{i+1}"] = tf.cast(tape1.gradient(globals()[f"ux{i+1}"], clps_group[i]), tf.float32)
                    globals()[f"ux{i+1}x{i+1}x{i+1}"] = tf.cast(tape1.gradient(globals()[f"ux{i+1}x{i+1}"], clps_group[i]), tf.float32)
            
            elif len(eqns) > 1:
                for e in range(len(eqns)):
                    globals()[f"u{e+1}"] = tf.cast(network(clps_group), tf.float32)[e]
                    for i in range(dim):
                        globals()[f"x{i+1}"] = tf.cast(clps_group[i], tf.float32)
                        globals()[f"u{e+1}x{i+1}"] = tf.cast(tape1.gradient(globals()[f"u{e+1}"], clps_group[i]), tf.float32)
                        globals()[f"u{e+1}x{i+1}x{i+1}"] = tf.cast(tape1.gradient(globals()[f"u{e+1}x{i+1}"], clps_group[i]), tf.float32)
                        globals()[f"u{e+1}x{i+1}x{i+1}x{i+1}"] = tf.cast(tape1.gradient(globals()[f"u{e+1}x{i+1}x{i+1}"], clps_group[i]), tf.float32)

        CLPloss = 0
        for e in range(len(eqns)):
            parse_tree = ast.parse(eqns[e], mode = "eval")
            eqnparse = eval(compile(parse_tree, "<string>", "eval"))
            # print(tf.reduce_mean(tf.square(eqnparse)))
            CLPloss += tf.reduce_mean(tf.square(eqnparse))
            CLPloss = tf.cast(CLPloss, tf.float32)

        BCloss = 0
        # periodic
        if bdry_type == 1:
            BCloss = tf.cast(0, tf.float32)
           
        # dirichlet
        elif bdry_type == 2:

            u_bc_pred = network(bcs_group)
            u_bcs = tf.cast(u_bcs, tf.float32)
            BCloss = tf.reduce_mean(tf.square(u_bcs-u_bc_pred))

        # neumann
        elif bdry_type == 3:
            with tf.GradientTape(persistent=True) as tape3:
                for col in bcs_group:
                    tape3.watch(col)
                u_bc_pred = network(bcs_group)

                tosort = bcs[:, 0]
                sorted = tf.argsort(tosort)
                bcs = tf.gather(bcs, sorted)
                for i in range(1, dim):
                    upper = bcs[:(2*i-1)*bdry_component_sizes]
                    lower = bcs[(np.shape(bcs)[0]-bdry_component_sizes):]
                    a = bcs[(2*i-1)*bdry_component_sizes:(np.shape(bcs)[0]-bdry_component_sizes)]
                    tosort = a[:, i]
                    sorted = tf.argsort(tosort)
                    a = tf.gather(a, sorted)
                    bcs = tf.concat([upper, lower, a], axis=0)
                u_bcs = bcs[:, -1:]

                for i in range(dim):
                    du_bc_pred = tf.cast(tape3.gradient(u_bc_pred, bcs_group[i]), tf.float32)
                    BCloss += tf.reduce_mean(tf.square(du_bc_pred[i*2*bdry_component_sizes:(i+1)*2*bdry_component_sizes] - 
                                                       tf.cast(u_bcs[i*2*bdry_component_sizes:(i+1)*2*bdry_component_sizes], tf.float32)))

        # ode
        elif bdry_type == 4:
            if boundary.get_flag() == "ic":
                t_orders = boundary.get_orders()
                t_ics = bcs[:,0:1]
                maxi = 0
                ics_group = [t_ics]
                for e in range(len(eqns)):
                    for i in range(maxi, maxi+t_orders[e]):
                        ics_group.append([bcs[:,i+1:i+2]])
                        maxi += 1
                with tf.GradientTape(persistent=True) as tape2:
                    for col in ics_group:
                        tape2.watch(col)
                    dim_iter = dim
                    for e in range(len(eqns)):
                        if (len(eqns) == 1):
                            u_init_pred = network(ics_group[:dim])
                        else:
                            u_init_pred = network(ics_group[:dim])[e]
                        BCloss += tf.reduce_mean(tf.square(u_init_pred - tf.cast(ics_group[dim_iter], tf.float32)))
                        dim_iter += 1
                        for i in range(t_orders[e]-1):
                            next_pred = tf.cast(tape2.gradient(u_init_pred, ics_group[0]), tf.float32)
                            BCloss += tf.reduce_mean(tf.square(next_pred - tf.cast(ics_group[(dim_iter)], tf.float32)))
                            u_init_pred = next_pred
                            dim_iter += 1

            elif boundary.get_flag() == "bc":
                pass

        loss = CLPloss + BCloss
    
    grads = tape.gradient(loss, network.trainable_variables)
    return CLPloss, BCloss, grads


def trainStepTime(eqns, clps, bcs, ics, network, boundary, t_orders):
    dim = boundary.get_domain().get_dim()
    bdry_type = boundary.get_bdry_type()
    bdry_components = boundary.get_domain().get_bdry_components()
    bdry_component_sizes = boundary.get_domain().get_bdry_component_size()

    t_clp = clps[:,0:1]
    clps_group = [t_clp]
    for i in range(dim):
        globals()[f"x{i+1}_clp"] = clps[:,i+1:i+2]
        clps_group.append(globals()[f"x{i+1}_clp"])
    
    t_ics = ics[:,0:1]
    maxi = 0
    ics_group = [t_ics]
    for i in range(dim):
        # globals()[f"x{i+1}_ics"] = ics[:,i+1:i+2]
        ics_group.append([ics[:,i+1:i+2]])
        maxi = i+1

    for e in range(len(eqns)):
        for i in range(maxi, maxi+t_orders[e]):
            # globals()[f"u{i}_ics"] = ics[:,i+1:i+2]
            ics_group.append([ics[:,i+1:i+2]])
            maxi += 1

    t_bcs = bcs[:,0:1]
    u_bcs = bcs[:,-1:]
    bcs_group = [t_bcs]
    for i in range(dim):
        globals()[f"x{i+1}_bc"] = bcs[:,i+1:i+2]
        bcs_group.append(globals()[f"x{i+1}_bc"])

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
            
            if (len(eqns)) == 1:
                u = tf.cast(network(clps_group), tf.float32)
                ut = tf.cast(tape1.gradient(u, clps_group[0]), tf.float32)
                utt = tf.cast(tape1.gradient(ut, clps_group[0]), tf.float32)
                uttt = tf.cast(tape1.gradient(utt, clps_group[0]), tf.float32)
                for i in range(dim):
                    globals()[f"x{i+1}"] = tf.cast(clps_group[i+1], tf.float32)
                    globals()[f"ux{i+1}"] = tf.cast(tape1.gradient(u, clps_group[i+1]), tf.float32)
                    globals()[f"ux{i+1}x{i+1}"] = tf.cast(tape1.gradient(globals()[f"ux{i+1}"], clps_group[i+1]), tf.float32)
                    globals()[f"ux{i+1}x{i+1}x{i+1}"] = tf.cast(tape1.gradient(globals()[f"ux{i+1}x{i+1}"], clps_group[i+1]), tf.float32)
            
            elif len(eqns) > 1:
                for e in range(len(eqns)):
                    globals()[f"u{e+1}"] = tf.cast(network(clps_group), tf.float32)[e]
                    globals()[f"u{e+1}t"] = tf.cast(tape1.gradient(globals()[f"u{e+1}"], clps_group[0]), tf.float32)
                    globals()[f"u{e+1}tt"] = tf.cast(tape1.gradient(globals()[f"u{e+1}t"], clps_group[0]), tf.float32)
                    globals()[f"u{e+1}ttt"] = tf.cast(tape1.gradient(globals()[f"u{e+1}tt"], clps_group[0]), tf.float32)
                    for i in range(dim):
                        globals()[f"x{i+1}"] = tf.cast(clps_group[i+1], tf.float32)
                        globals()[f"u{e+1}x{i+1}"] = tf.cast(tape1.gradient(globals()[f"u{e+1}"], clps_group[i+1]), tf.float32)
                        globals()[f"u{e+1}x{i+1}x{i+1}"] = tf.cast(tape1.gradient(globals()[f"u{e+1}x{i+1}"], clps_group[i+1]), tf.float32)
                        globals()[f"u{e+1}x{i+1}x{i+1}x{i+1}"] = tf.cast(tape1.gradient(globals()[f"u{e+1}x{i+1}x{i+1}"], clps_group[i+1]), tf.float32)


        t = tf.cast(clps_group[0], tf.float32)

        CLPloss = 0
        for e in range(len(eqns)):
            parse_tree = ast.parse(eqns[e], mode = "eval")
            eqnparse = eval(compile(parse_tree, "<string>", "eval"))
            # print(tf.reduce_mean(tf.square(eqnparse)))
            CLPloss += tf.reduce_mean(tf.square(eqnparse))
            CLPloss = tf.cast(CLPloss, tf.float32)
        
        BCloss = 0

        # periodic
        if bdry_type == 1:
            BCloss = tf.cast(0, tf.float32)
            
        # dirichlet
        elif bdry_type == 2:
            u_bc_pred = network(bcs_group)
            u_bcs = tf.cast(u_bcs, tf.float32)
            BCloss = tf.reduce_mean(tf.square(u_bcs-u_bc_pred))

        # neumann
        elif bdry_type == 3:
            with tf.GradientTape(persistent=True) as tape3:
                for col in bcs_group:
                    tape3.watch(col)
                u_bc_pred = network(bcs_group)

                tosort = bcs[:, 1]
                sorted = tf.argsort(tosort)
                bcs = tf.gather(bcs, sorted)
                for i in range(1, dim):
                    upper = bcs[:(2*i-1)*bdry_component_sizes]
                    lower = bcs[(np.shape(bcs)[0]-bdry_component_sizes):]
                    a = bcs[(2*i-1)*bdry_component_sizes:(np.shape(bcs)[0]-bdry_component_sizes)]
                    tosort = a[:, i+1]
                    sorted = tf.argsort(tosort)
                    a = tf.gather(a, sorted)
                    bcs = tf.concat([upper, lower, a], axis=0)
                u_bcs = bcs[:, -1:]

                for i in range(dim):
                    du_bc_pred = tf.cast(tape3.gradient(u_bc_pred, bcs_group[i+1]), tf.float32)
                    BCloss += tf.reduce_mean(tf.square(du_bc_pred[i*2*bdry_component_sizes:(i+1)*2*bdry_component_sizes] - 
                                                       tf.cast(u_bcs[i*2*bdry_component_sizes:(i+1)*2*bdry_component_sizes], tf.float32)))
                    
        # ode
        elif bdry_type == 4:
            pass
    

        ICloss = 0
        with tf.GradientTape(persistent=True) as tape2:
            for col in ics_group:
                tape2.watch(col)

            dim_iter = dim

            for e in range(len(eqns)):
                if (len(eqns) == 1):
                    u_init_pred = network(ics_group[:dim+1])
                else:
                    u_init_pred = network(ics_group[:dim+1])[e]

                ICloss += tf.reduce_mean(tf.square(u_init_pred - tf.cast(ics_group[dim_iter+1], tf.float32)))
                dim_iter += 1

                for i in range(t_orders[e]-1):
                    next_pred = tf.cast(tape2.gradient(u_init_pred, ics_group[0]), tf.float32)
                    ICloss += tf.reduce_mean(tf.square(next_pred - tf.cast(ics_group[(dim_iter+1)], tf.float32)))
                    u_init_pred = next_pred
                    dim_iter += 1
        
    
        loss = CLPloss + BCloss + ICloss
    
    grads = tape.gradient(loss, network.trainable_variables)
    return CLPloss, BCloss, ICloss, grads