import jax
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import ast

def timeStep_order1(steps, tfinal, deeponet, params, num_points, init_data, t):
    tfinalsteps = tfinal*steps

    tt = np.linspace(0, tfinalsteps, (num_points-1)*steps)
    
    zinit = init_data

    zall = []

    for i in range(steps):
        zinit = np.repeat(np.expand_dims(zinit, axis=0), len(t), axis=0)

        u = deeponet.apply(params, t, zinit[:,:1])

        zinit = np.array([u[-1]])
        zall.append(u[:-1])

    uall = np.concatenate(zall, axis=0)

    plt.plot(tt, uall)
    plt.title("Time-stepped neural network")
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('u')
    plt.savefig("ODE-TimeStep-Pred")
    plt.clf()

    return


def timeStep_order2(steps, tfinal, deeponet, params, num_points, init_data, t):
    tfinalsteps = tfinal*steps
    tt = np.linspace(0, tfinalsteps, (num_points-1)*steps)
    @jax.jit
    def u_model(t, z, zt, params):
        return deeponet.apply(params, t, z, zt)[0]
    @jax.jit
    def u_t(t, z, zt, params):
        return jax.vmap(jax.grad(u_model, 0), [0, 0, 0, None])(t, z, zt, params)
    
    zinit = init_data

    zall = []

    for i in range(steps):
        zinit = np.repeat(np.expand_dims(zinit, axis=0), len(t), axis=0)

        u = deeponet.apply(params, t, zinit[:,:1], zinit[:,1:])
        ut = u_t(t, zinit[:,:1], zinit[:,1:], params)


        zinit = np.array([u[-1], ut[-1]])
        zall.append(u[:-1])

    uall = np.concatenate(zall, axis=0)

    plt.plot(tt, uall)
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('u')
    plt.savefig("ODE-TimeStep-Pred")
    plt.clf()

    return

def timeStep_order3(steps, tfinal, deeponet, params, num_points, init_data, t):
    tfinalsteps = tfinal*steps
    tt = np.linspace(0, tfinalsteps, (num_points-1)*steps)
    @jax.jit
    def u_model(t, z, zt, ztt, params):
        return deeponet.apply(params, t, z, zt, ztt)[0]
    @jax.jit
    def u_t(t, z, zt, ztt, params):
        return jax.vmap(jax.grad(u_model, 0), [0, 0, 0, 0, None])(t, z, zt, ztt, params)
    @jax.jit
    def u_tt(t, z, zt, ztt, params):
        return jax.vmap(jax.grad(jax.grad(u_model, 0),0), [0, 0, 0, 0, None])(t, z, zt, ztt, params)
    
    zinit = init_data

    zall = []

    for i in range(steps):
        zinit = np.repeat(np.expand_dims(zinit, axis=0), len(t), axis=0)

        u = deeponet.apply(params, t, zinit[:,:1], zinit[:,1:2], zinit[:,2:])
        ut = u_t(t, zinit[:,:1], zinit[:,1:2], zinit[:,2:], params)
        utt = u_tt(t, zinit[:,:1], zinit[:,1:2], zinit[:,2:], params)


        zinit = np.array([u[-1], ut[-1], utt[-1]])
        zall.append(u[:-1])

    uall = np.concatenate(zall, axis=0)

    plt.plot(tt, uall)
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('u')
    plt.savefig("ODE-TimeStep-Pred")
    plt.clf()
    
    return


def timeStep_SystemSelect(steps, t_bdry, model, params, N_pde, inits, t, order, original_orders, fileTitle):
    if len(order) == 2:
        if (order[0] == 1 and order[1] == 1):
            return timeStep_orders11(steps, t_bdry, model, params, N_pde, inits, t, original_orders, fileTitle)
        elif (order[0] == 2 and order[1] == 1):
            return timeStep_orders21(steps, t_bdry, model, params, N_pde, inits, t, original_orders, fileTitle)
        elif (order[0] == 3 and order[1] == 1):
            return timeStep_orders31(steps, t_bdry, model, params, N_pde, inits, t, original_orders, fileTitle)
        elif (order[0] == 2 and order[1] == 2):
            return timeStep_orders22(steps, t_bdry, model, params, N_pde, inits, t, original_orders, fileTitle)
        elif (order[0] == 3 and order[1] == 2):
            return timeStep_orders32(steps, t_bdry, model, params, N_pde, inits, t, original_orders, fileTitle)
        elif(order[0] == 3 and order[1] == 3):
            return timeStep_orders33(steps, t_bdry, model, params, N_pde, inits, t, original_orders, fileTitle)
    elif len(order) == 3:
        if (order[0] == 1 and order[1] == 1 and order[2] == 1):
            return timeStep_orders111(steps, t_bdry, model, params, N_pde, inits, t, original_orders, fileTitle)
        elif (order[0] == 2 and order[1] == 1 and order[2] == 1):
            return timeStep_orders211(steps, t_bdry, model, params, N_pde, inits, t, original_orders, fileTitle)
        elif (order[0] == 2 and order[1] == 2 and order[2] == 1):
            return timeStep_orders221(steps, t_bdry, model, params, N_pde, inits, t, original_orders, fileTitle)
        elif (order[0] == 2 and order[1] == 2 and order[2] == 2):
            return timeStep_orders222(steps, t_bdry, model, params, N_pde, inits, t, original_orders, fileTitle)
        elif (order[0] == 3 and order[1] == 1 and order[2] == 1):
            return timeStep_orders311(steps, t_bdry, model, params, N_pde, inits, t, original_orders, fileTitle)
        elif (order[0] == 3 and order[1] == 2 and order[2] == 1):
            return timeStep_orders321(steps, t_bdry, model, params, N_pde, inits, t, original_orders, fileTitle)
        elif (order[0] == 3 and order[1] == 3 and order[2] == 1):
            return timeStep_orders331(steps, t_bdry, model, params, N_pde, inits, t, original_orders, fileTitle)
        elif (order[0] == 3 and order[1] == 3 and order[2] == 2):
            return timeStep_orders332(steps, t_bdry, model, params, N_pde, inits, t, original_orders, fileTitle)
        elif (order[0] == 3 and order[1] == 3 and order[2] == 3):
            return timeStep_orders333(steps, t_bdry, model, params, N_pde, inits, t, original_orders, fileTitle)
    return

def timeStep_orders11(steps, tfinal, deeponet, params, num_points, init_data, t, orig_orders, title):
    tfinalsteps = tfinal*steps
    tt = np.linspace(0, tfinalsteps, (num_points-1)*steps)
    
    zall = []
    zinit = []
    for i in init_data:
        for j in i:
            zinit.append(j)

    for i in range(steps):
        zinit = np.repeat(np.expand_dims(zinit, axis=0), len(t), axis=0)

        u, x = deeponet.apply(params, t, zinit)

        zinit = np.array([u[-1], x[-1]])
        zall.append(np.column_stack([u[:-1], x[:-1]]))
    uall = np.concatenate(zall, axis=0)


    plt.plot(tt, uall[:,0])
    plt.plot(tt, uall[:,1])
    plt.grid()
    plt.legend(["u(t)", "x(t)"])
    plt.xlabel('t')
    plt.ylabel('u, x')
    plt.savefig(title)
    plt.clf()

    return

def timeStep_orders21(steps, tfinal, deeponet, params, num_points, init_data, t, orig_orders, title):
    tfinalsteps = tfinal*steps
    tt = np.linspace(0, tfinalsteps, (num_points-1)*steps)
    @partial(jax.jit, static_argnames=["component"])
    def w_model(t, z, component, params):
        return deeponet.apply(params, t, z)[component][0]
    @partial(jax.jit, static_argnames=["component"])
    def w_t(t, z, component, params):
        return jax.vmap(jax.grad(w_model, 0), [0, 0, None, None])(t, z, component, params)
    
    zall = []
    zinit = []
    for i in init_data:
        for j in i:
            zinit.append(j)

    for i in range(steps):
        zinit = np.repeat(np.expand_dims(zinit, axis=0), len(t), axis=0)

        x1, x2 = deeponet.apply(params, t, zinit)

        x1t = w_t(t, zinit, 0, params)
        
        zinit = np.array([x1[-1], x1t[-1], x2[-1]])
        zall.append(np.column_stack([x1[:-1], x2[:-1]]))
    uall = np.concatenate(zall, axis=0)

    # exact_outs = []
    # exact_eqn = ["np.sin(t)+0.5*np.cos(t)", "-0.5*np.sin(t)+np.cos(t)+1"]
    # for j in range(2):
    #     exact_output = []
    #     for i in tt:
    #         t = i
    #         parse_tree = ast.parse(exact_eqn[j], mode="eval")
    #         eqn = eval(compile(parse_tree, "<string>", "eval"))
    #         exact_output.append(eqn)
    #     exact_outs.append(exact_output)

    plt.plot(tt, uall[:,0])
    plt.plot(tt, uall[:,1])

    # plt.plot(tt, exact_outs[0], "--")
    # plt.plot(tt, exact_outs[1], "--")
    #plt.title("Neural network against exact solutions")

    plt.title("Time-stepped neural network")
    plt.grid()
    if (orig_orders[0] == 2 and orig_orders[1] == 1):
        plt.legend(["u(t)", "x(t)"])
        #plt.legend(["u(t)", "x(t)", "Exact u(t)", "Exact x(t)"])
    elif (orig_orders[0] == 1 and orig_orders[1] == 2):
        plt.legend(["x(t)", "x(t)"])
    plt.xlabel('t')
    plt.ylabel('u, x')
    plt.savefig(title)
    plt.clf()

    # plt.plot(tt, uall[:,0] - exact_outs[0])
    # plt.plot(tt, uall[:,1] - exact_outs[1])
    # plt.title("Time-stepped error")
    # plt.legend(["u(t) error", "v(t) error"])
    # plt.xlabel("t")
    # plt.ylabel("error")
    # plt.grid()
    # plt.savefig("ODE-TimeStep-Error")
    # plt.clf()

    return

def timeStep_orders31(steps, tfinal, deeponet, params, num_points, init_data, t, orig_orders, title):
    tfinalsteps = tfinal*steps
    tt = np.linspace(0, tfinalsteps, (num_points-1)*steps)
    @partial(jax.jit, static_argnames=["component"])
    def w_model(t, z, component, params):
        return deeponet.apply(params, t, z)[component][0]
    @partial(jax.jit, static_argnames=["component"])
    def w_t(t, z, component, params):
        return jax.vmap(jax.grad(w_model, 0), [0, 0, None, None])(t, z, component, params)
    @partial(jax.jit, static_argnames=["component"])
    def w_tt(t, z, component, params):
        return jax.vmap(jax.grad(jax.grad(w_model, 0),0), [0, 0, None, None])(t, z, component, params)
    
    zall = []
    zinit = []
    for i in init_data:
        for j in i:
            zinit.append(j)

    for i in range(steps):
        zinit = np.repeat(np.expand_dims(zinit, axis=0), len(t), axis=0)

        x1, x2 = deeponet.apply(params, t, zinit)

        x1t = w_t(t, zinit, 0, params)
        x1tt = w_tt(t, zinit, 0, params)
       
        zinit = np.array([x1[-1], x1t[-1], x1tt[-1], x2[-1]])
        zall.append(np.column_stack([x1[:-1], x2[:-1]]))
    uall = np.concatenate(zall, axis=0)

    plt.plot(tt, uall[:,0])
    plt.plot(tt, uall[:,1])
    plt.grid()
    if (orig_orders[0] == 3 and orig_orders[1] == 1):
        plt.legend(["u(t)", "x(t)"])
    elif (orig_orders[0] == 1 and orig_orders[1] == 3):
        plt.legend(["x(t)", "u(t)"])
    plt.xlabel('t')
    plt.ylabel('u, x')
    plt.savefig(title)
    plt.clf()

    return

def timeStep_orders22(steps, tfinal, deeponet, params, num_points, init_data, t, orig_orders, title):
    tfinalsteps = tfinal*steps
    tt = np.linspace(0, tfinalsteps, (num_points-1)*steps)
    @partial(jax.jit, static_argnames=["component"])
    def w_model(t, z, component, params):
        return deeponet.apply(params, t, z)[component][0]
    @partial(jax.jit, static_argnames=["component"])
    def w_t(t, z, component, params):
        return jax.vmap(jax.grad(w_model, 0), [0, 0, None, None])(t, z, component, params)
    
    zall = []
    zinit = []
    for i in init_data:
        for j in i:
            zinit.append(j)

    for i in range(steps):
        zinit = np.repeat(np.expand_dims(zinit, axis=0), len(t), axis=0)

        x1, x2 = deeponet.apply(params, t, zinit)

        x1t = w_t(t, zinit, 0, params)
        x2t = w_t(t, zinit, 1, params)
        
        zinit = np.array([x1[-1], x1t[-1], x2[-1], x2t[-1]])
        zall.append(np.column_stack([x1[:-1], x2[:-1]]))
    uall = np.concatenate(zall, axis=0)

    plt.plot(tt, uall[:,0])
    plt.plot(tt, uall[:,1])
    plt.grid()
    plt.legend(["u(t)", "x(t)"])
    plt.ylabel('u, x')
    plt.savefig(title)
    plt.clf()

    return

def timeStep_orders32(steps, tfinal, deeponet, params, num_points, init_data, t, orig_orders, title):
    tfinalsteps = tfinal*steps
    tt = np.linspace(0, tfinalsteps, (num_points-1)*steps)
    @partial(jax.jit, static_argnames=["component"])
    def w_model(t, z, component, params):
        return deeponet.apply(params, t, z)[component][0]
    @partial(jax.jit, static_argnames=["component"])
    def w_t(t, z, component, params):
        return jax.vmap(jax.grad(w_model, 0), [0, 0, None, None])(t, z, component, params)
    @partial(jax.jit, static_argnames=["component"])
    def w_tt(t, z, component, params):
        return jax.vmap(jax.grad(jax.grad(w_model, 0),0), [0, 0, None, None])(t, z, component, params)
    
    zall = []
    zinit = []
    for i in init_data:
        for j in i:
            zinit.append(j)

    for i in range(steps):
        zinit = np.repeat(np.expand_dims(zinit, axis=0), len(t), axis=0)

        x1, x2 = deeponet.apply(params, t, zinit)

        x1t = w_t(t, zinit, 0, params)
        x1tt = w_tt(t, zinit, 0, params)
        x2t = w_t(t, zinit, 1, params)
       
        zinit = np.array([x1[-1], x1t[-1], x1tt[-1], x2[-1], x2t[-1]])
        zall.append(np.column_stack([x1[:-1], x2[:-1]]))
    uall = np.concatenate(zall, axis=0)

    plt.plot(tt, uall[:,0])
    plt.plot(tt, uall[:,1])
    plt.grid()
    if (orig_orders[0] == 3 and orig_orders[1] == 2):
        plt.legend(["u(t)", "x(t)"])
    elif (orig_orders[0] == 2 and orig_orders[1] == 3):
        plt.legend(["x(t)", "u(t)"])
    plt.xlabel('t')
    plt.ylabel('u, x')
    plt.savefig(title)
    plt.clf()

    return

def timeStep_orders33(steps, tfinal, deeponet, params, num_points, init_data, t, orig_orders, title):
    tfinalsteps = tfinal*steps
    tt = np.linspace(0, tfinalsteps, (num_points-1)*steps)
    @partial(jax.jit, static_argnames=["component"])
    def w_model(t, z, component, params):
        return deeponet.apply(params, t, z)[component][0]
    @partial(jax.jit, static_argnames=["component"])
    def w_t(t, z, component, params):
        return jax.vmap(jax.grad(w_model, 0), [0, 0, None, None])(t, z, component, params)
    @partial(jax.jit, static_argnames=["component"])
    def w_tt(t, z, component, params):
        return jax.vmap(jax.grad(jax.grad(w_model, 0),0), [0, 0, None, None])(t, z, component, params)
    
    zall = []
    zinit = []
    for i in init_data:
        for j in i:
            zinit.append(j)

    for i in range(steps):
        zinit = np.repeat(np.expand_dims(zinit, axis=0), len(t), axis=0)

        x1, x2 = deeponet.apply(params, t, zinit)

        x1t = w_t(t, zinit, 0, params)
        x1tt = w_tt(t, zinit, 0, params)
        x2t = w_t(t, zinit, 1, params)
        x2tt = w_tt(t, zinit, 1, params)
       
        zinit = np.array([x1[-1], x1t[-1], x1tt[-1], x2[-1], x2t[-1], x2tt[-1]])
        zall.append(np.column_stack([x1[:-1], x2[:-1]]))
    uall = np.concatenate(zall, axis=0)

    plt.plot(tt, uall[:,0])
    plt.plot(tt, uall[:,1])
    plt.grid()
    plt.legend(["u(t)", "x(t)"])
    plt.xlabel('t')
    plt.ylabel('u, x')
    plt.savefig(title)
    plt.clf()

    return


def timeStep_orders111(steps, tfinal, deeponet, params, num_points, init_data, t, orig_orders, title):
    tfinalsteps = tfinal*steps
    tt = np.linspace(0, tfinalsteps, (num_points-1)*steps)

    zall = []
    zinit = []
    for i in init_data:
        for j in i:
            zinit.append(j)

    for i in range(steps):
        zinit = np.repeat(np.expand_dims(zinit, axis=0), len(t), axis=0)

        u, x, y = deeponet.apply(params, t, zinit)

        zinit = np.array([u[-1], x[-1], y[-1]])
        zall.append(np.column_stack([u[:-1], x[:-1], y[:-1]]))
    uall = np.concatenate(zall, axis=0)

    plt.plot(tt, uall[:,0])
    plt.plot(tt, uall[:,1])
    plt.plot(tt, uall[:,2])
    plt.grid()
    plt.legend(["u(t)", "x(t)", "y(t)"])
    plt.xlabel('t')
    plt.ylabel('u, x, y')
    plt.savefig(title)
    plt.clf()

    return

def timeStep_orders211(steps, tfinal, deeponet, params, num_points, init_data, t, orig_orders, title):
    tfinalsteps = tfinal*steps
    tt = np.linspace(0, tfinalsteps, (num_points-1)*steps)
    @partial(jax.jit, static_argnames=["component"])
    def w_model(t, z, component, params):
        return deeponet.apply(params, t, z)[component][0]
    @partial(jax.jit, static_argnames=["component"])
    def w_t(t, z, component, params):
        return jax.vmap(jax.grad(w_model, 0), [0, 0, None, None])(t, z, component, params)

    zall = []
    zinit = []
    for i in init_data:
        for j in i:
            zinit.append(j)

    for i in range(steps):
        zinit = np.repeat(np.expand_dims(zinit, axis=0), len(t), axis=0)

        x1, x2, x3 = deeponet.apply(params, t, zinit)
        x1t = w_t(t, zinit, 0, params)

        zinit = np.array([x1[-1], x1t[-1], x2[-1], x3[-1]])
        zall.append(np.column_stack([x1[:-1] ,x2[:-1], x3[:-1]]))
    uall = np.concatenate(zall, axis=0)

    plt.plot(tt, uall[:,0])
    plt.plot(tt, uall[:,1])
    plt.plot(tt, uall[:,2])
    plt.grid()
    if (orig_orders[0] == 2 and orig_orders[1] == 1 and orig_orders[2] == 1):
        plt.legend(["u(t)", "x(t)", "y(t)"])
    elif (orig_orders[0] == 1 and orig_orders[1] == 2 and orig_orders[2] == 1):
        plt.legend(["x(t)", "u(t)", "y(t)"])
    elif (orig_orders[0] == 1 and orig_orders[1] == 1 and orig_orders[2] == 2):
        plt.legend(["y(t)", "u(t)", "x(t)"])
    plt.xlabel('t')
    plt.ylabel('u, x, y')
    plt.savefig(title)
    plt.clf()

    return

def timeStep_orders221(steps, tfinal, deeponet, params, num_points, init_data, t, orig_orders, title):
    tfinalsteps = tfinal*steps
    tt = np.linspace(0, tfinalsteps, (num_points-1)*steps)
    @partial(jax.jit, static_argnames=["component"])
    def w_model(t, z, component, params):
        return deeponet.apply(params, t, z)[component][0]
    @partial(jax.jit, static_argnames=["component"])
    def w_t(t, z, component, params):
        return jax.vmap(jax.grad(w_model, 0), [0, 0, None, None])(t, z, component, params)

    zall = []
    zinit = []
    for i in init_data:
        for j in i:
            zinit.append(j)

    for i in range(steps):
        zinit = np.repeat(np.expand_dims(zinit, axis=0), len(t), axis=0)

        x1, x2, x3 = deeponet.apply(params, t, zinit)
        x1t = w_t(t, zinit, 0, params)
        x2t = w_t(t, zinit, 1, params)

        zinit = np.array([x1[-1], x1t[-1], x2[-1], x2t[-1], x3[-1]])
        zall.append(np.column_stack([x1[:-1] ,x2[:-1], x3[:-1]]))
    uall = np.concatenate(zall, axis=0)

    plt.plot(tt, uall[:,0])
    plt.plot(tt, uall[:,1])
    plt.plot(tt, uall[:,2])
    plt.grid()
    if (orig_orders[0] == 2 and orig_orders[1] == 2 and orig_orders[2] == 1):
        plt.legend(["u(t)", "x(t)", "y(t)"])
    elif (orig_orders[0] == 2 and orig_orders[1] == 1 and orig_orders[2] == 2):
        plt.legend(["u(t)", "y(t)", "x(t)"])
    elif (orig_orders[0] == 1 and orig_orders[1] == 2 and orig_orders[2] == 2):
        plt.legend(["x(t)", "y(t)", "u(t)"])
    plt.xlabel('t')
    plt.ylabel('u, x, y')
    plt.savefig(title)
    plt.clf()

    return

def timeStep_orders222(steps, tfinal, deeponet, params, num_points, init_data, t, orig_orders, title):
    tfinalsteps = tfinal*steps
    tt = np.linspace(0, tfinalsteps, (num_points-1)*steps)
    @partial(jax.jit, static_argnames=["component"])
    def w_model(t, z, component, params):
        return deeponet.apply(params, t, z)[component][0]
    @partial(jax.jit, static_argnames=["component"])
    def w_t(t, z, component, params):
        return jax.vmap(jax.grad(w_model, 0), [0, 0, None, None])(t, z, component, params)

    zall = []
    zinit = []
    for i in init_data:
        for j in i:
            zinit.append(j)

    for i in range(steps):
        zinit = np.repeat(np.expand_dims(zinit, axis=0), len(t), axis=0)

        x1, x2, x3 = deeponet.apply(params, t, zinit)
        x1t = w_t(t, zinit, 0, params)
        x2t = w_t(t, zinit, 1, params)
        x3t = w_t(t, zinit, 2, params)

        zinit = np.array([x1[-1], x1t[-1], x2[-1], x2t[-1], x3[-1], x3t[-1]])
        zall.append(np.column_stack([x1[:-1] ,x2[:-1], x3[:-1]]))
    uall = np.concatenate(zall, axis=0)

    plt.plot(tt, uall[:,0])
    plt.plot(tt, uall[:,1])
    plt.plot(tt, uall[:,2])
    plt.grid()
    plt.legend(["u(t)", "x(t)", "y(t)"])
    plt.xlabel('t')
    plt.ylabel('u, x, y')
    plt.savefig(title)
    plt.clf()

    return

def timeStep_orders311(steps, tfinal, deeponet, params, num_points, init_data, t, orig_orders, title):
    tfinalsteps = tfinal*steps
    tt = np.linspace(0, tfinalsteps, (num_points-1)*steps)
    @partial(jax.jit, static_argnames=["component"])
    def w_model(t, z, component, params):
        return deeponet.apply(params, t, z)[component][0]
    @partial(jax.jit, static_argnames=["component"])
    def w_t(t, z, component, params):
        return jax.vmap(jax.grad(w_model, 0), [0, 0, None, None])(t, z, component, params)
    @partial(jax.jit, static_argnames=["component"])
    def w_tt(t, z, component, params):
        return jax.vmap(jax.grad(jax.grad(w_model, 0),0), [0, 0, None, None])(t, z, component, params)

    zall = []
    zinit = []
    for i in init_data:
        for j in i:
            zinit.append(j)

    for i in range(steps):
        zinit = np.repeat(np.expand_dims(zinit, axis=0), len(t), axis=0)

        x1, x2, x3 = deeponet.apply(params, t, zinit)
        x1t = w_t(t, zinit, 0, params)
        x1tt = w_tt(t, zinit, 0, params)

        zinit = np.array([x1[-1], x1t[-1], x1tt[-1], x2[-1], x3[-1]])
        zall.append(np.column_stack([x1[:-1] ,x2[:-1], x3[:-1]]))
    uall = np.concatenate(zall, axis=0)

    plt.plot(tt, uall[:,0])
    plt.plot(tt, uall[:,1])
    plt.plot(tt, uall[:,2])
    plt.grid()
    if (orig_orders[0] == 3 and orig_orders[1] == 1 and orig_orders[2] == 1):
        plt.legend(["u(t)", "x(t)", "y(t)"])
    elif (orig_orders[0] == 1 and orig_orders[1] == 3 and orig_orders[2] == 1):
        plt.legend(["x(t)", "u(t)", "y(t)"])
    elif (orig_orders[0] == 1 and orig_orders[1] == 1 and orig_orders[2] == 3):
        plt.legend(["y(t)", "u(t)", "x(t)"])
    plt.xlabel('t')
    plt.ylabel('u, x, y')
    plt.savefig(title)
    plt.clf()

    return

def timeStep_orders321(steps, tfinal, deeponet, params, num_points, init_data, t, orig_orders, title):
    tfinalsteps = tfinal*steps
    tt = np.linspace(0, tfinalsteps, (num_points-1)*steps)
    @partial(jax.jit, static_argnames=["component"])
    def w_model(t, z, component, params):
        return deeponet.apply(params, t, z)[component][0]
    @partial(jax.jit, static_argnames=["component"])
    def w_t(t, z, component, params):
        return jax.vmap(jax.grad(w_model, 0), [0, 0, None, None])(t, z, component, params)
    @partial(jax.jit, static_argnames=["component"])
    def w_tt(t, z, component, params):
        return jax.vmap(jax.grad(jax.grad(w_model, 0),0), [0, 0, None, None])(t, z, component, params)

    zall = []
    zinit = []
    for i in init_data:
        for j in i:
            zinit.append(j)

    for i in range(steps):
        zinit = np.repeat(np.expand_dims(zinit, axis=0), len(t), axis=0)

        x1, x2, x3 = deeponet.apply(params, t, zinit)
        x1t = w_t(t, zinit, 0, params)
        x1tt = w_tt(t, zinit, 0, params)
        x2t = w_t(t, zinit, 1, params)

        zinit = np.array([x1[-1], x1t[-1], x1tt[-1], x2[-1], x2t[-1], x3[-1]])
        zall.append(np.column_stack([x1[:-1] ,x2[:-1], x3[:-1]]))
    uall = np.concatenate(zall, axis=0)

    plt.plot(tt, uall[:,0])
    plt.plot(tt, uall[:,1])
    plt.plot(tt, uall[:,2])
    plt.grid()
    if ((orig_orders[0] == 3) and (orig_orders[1] == 2) and (orig_orders[2] == 1)):
        plt.legend(["u(t)", "x(t)", "y(t)"]) #u, x, y -> u, x, y
    elif ((orig_orders[0] == 3) and (orig_orders[1] == 1) and (orig_orders[2] == 2)):
        plt.legend(["u(t)", "y(t)", "x(t)"]) #u, x, y -> u, y, x
    elif ((orig_orders[0] == 2) and (orig_orders[1] == 1) and (orig_orders[2] == 3)):
        plt.legend(["y(t)", "u(t)", "x(t)"]) #u, x, y -> y, u, x
    elif ((orig_orders[0] == 2) and (orig_orders[1] == 3) and (orig_orders[2] == 1)):
        plt.legend(["x(t)", "u(t)", "y(t)"]) #u, x, y -> x, u, y
    elif ((orig_orders[0] == 1) and (orig_orders[1] == 2) and (orig_orders[2] == 3)):
        plt.legend(["y(t)", "x(t)", "u(t)"]) #u, x, y -> y, x, u
    elif ((orig_orders[0] == 1) and (orig_orders[1] == 3) and (orig_orders[2] == 2)):
        plt.legend(["x(t)", "y(t)", "u(t)"]) #y, x, y -> x, y, u
    plt.xlabel('t')
    plt.ylabel('u, x, y')
    plt.savefig(title)
    plt.clf()

    return

def timeStep_orders331(steps, tfinal, deeponet, params, num_points, init_data, t, orig_orders, title):
    tfinalsteps = tfinal*steps
    tt = np.linspace(0, tfinalsteps, (num_points-1)*steps)
    @partial(jax.jit, static_argnames=["component"])
    def w_model(t, z, component, params):
        return deeponet.apply(params, t, z)[component][0]
    @partial(jax.jit, static_argnames=["component"])
    def w_t(t, z, component, params):
        return jax.vmap(jax.grad(w_model, 0), [0, 0, None, None])(t, z, component, params)
    @partial(jax.jit, static_argnames=["component"])
    def w_tt(t, z, component, params):
        return jax.vmap(jax.grad(jax.grad(w_model, 0),0), [0, 0, None, None])(t, z, component, params)

    zall = []
    zinit = []
    for i in init_data:
        for j in i:
            zinit.append(j)

    for i in range(steps):
        zinit = np.repeat(np.expand_dims(zinit, axis=0), len(t), axis=0)

        x1, x2, x3 = deeponet.apply(params, t, zinit)
        x1t = w_t(t, zinit, 0, params)
        x1tt = w_tt(t, zinit, 0, params)
        x2t = w_t(t, zinit, 1, params)
        x2tt = w_tt(t, zinit, 1, params)

        zinit = np.array([x1[-1], x1t[-1], x1tt[-1], x2[-1], x2t[-1], x2tt[-1], x3[-1]])
        zall.append(np.column_stack([x1[:-1] ,x2[:-1], x3[:-1]]))
    uall = np.concatenate(zall, axis=0)

    plt.plot(tt, uall[:,0])
    plt.plot(tt, uall[:,1])
    plt.plot(tt, uall[:,2])
    plt.grid()
    if ((orig_orders[0] == 3) and (orig_orders[1] == 3) and (orig_orders[2] == 1)):
        plt.legend(["u(t)", "x(t)", "y(t)"]) 
    elif ((orig_orders[0] == 3) and (orig_orders[1] == 1) and (orig_orders[2] == 3)):
        plt.legend(["u(t)", "y(t)", "x(t)"]) 
    elif ((orig_orders[0] == 1) and (orig_orders[1] == 3) and (orig_orders[2] == 3)):
        plt.legend(["x(t)", "y(t)", "u(t)"]) 
    plt.xlabel('t')
    plt.ylabel('u, x, y')
    plt.savefig(title)
    plt.clf()

    return

def timeStep_orders332(steps, tfinal, deeponet, params, num_points, init_data, t, orig_orders, title):
    tfinalsteps = tfinal*steps
    tt = np.linspace(0, tfinalsteps, (num_points-1)*steps)
    @partial(jax.jit, static_argnames=["component"])
    def w_model(t, z, component, params):
        return deeponet.apply(params, t, z)[component][0]
    @partial(jax.jit, static_argnames=["component"])
    def w_t(t, z, component, params):
        return jax.vmap(jax.grad(w_model, 0), [0, 0, None, None])(t, z, component, params)
    @partial(jax.jit, static_argnames=["component"])
    def w_tt(t, z, component, params):
        return jax.vmap(jax.grad(jax.grad(w_model, 0),0), [0, 0, None, None])(t, z, component, params)

    zall = []
    zinit = []
    for i in init_data:
        for j in i:
            zinit.append(j)

    for i in range(steps):
        zinit = np.repeat(np.expand_dims(zinit, axis=0), len(t), axis=0)

        x1, x2, x3 = deeponet.apply(params, t, zinit)

        x1t = w_t(t, zinit, 0, params)
        x1tt = w_tt(t, zinit, 0, params)
        x2t = w_t(t, zinit, 1, params)
        x2tt = w_tt(t, zinit, 1, params)
        x3t = w_t(t, zinit, 2, params)

        zinit = np.array([x1[-1], x1t[-1], x1tt[-1], x2[-1], x2t[-1], x2tt[-1], x3[-1], x3t[-1]])
        zall.append(np.column_stack([x1[:-1] ,x2[:-1], x3[:-1]]))
    uall = np.concatenate(zall, axis=0)

    plt.plot(tt, uall[:,0])
    plt.plot(tt, uall[:,1])
    plt.plot(tt, uall[:,2])
    plt.grid()
    if ((orig_orders[0] == 3) and (orig_orders[1] == 3) and (orig_orders[2] == 2)):
        plt.legend(["u(t)", "x(t)", "y(t)"]) 
    elif ((orig_orders[0] == 3) and (orig_orders[1] == 2) and (orig_orders[2] == 3)):
        plt.legend(["u(t)", "y(t)", "x(t)"]) 
    elif ((orig_orders[0] == 2) and (orig_orders[1] == 3) and (orig_orders[2] == 3)):
        plt.legend(["x(t)", "y(t)", "u(t)"]) 
    plt.xlabel('t')
    plt.ylabel('u, x, y')
    plt.savefig(title)
    plt.clf()

    return

def timeStep_orders333(steps, tfinal, deeponet, params, num_points, init_data, t, orig_orders, title):
    tfinalsteps = tfinal*steps
    tt = np.linspace(0, tfinalsteps, (num_points-1)*steps)
    @partial(jax.jit, static_argnames=["component"])
    def w_model(t, z, component, params):
        return deeponet.apply(params, t, z)[component][0]
    @partial(jax.jit, static_argnames=["component"])
    def w_t(t, z, component, params):
        return jax.vmap(jax.grad(w_model, 0), [0, 0, None, None])(t, z, component, params)
    @partial(jax.jit, static_argnames=["component"])
    def w_tt(t, z, component, params):
        return jax.vmap(jax.grad(jax.grad(w_model, 0),0), [0, 0, None, None])(t, z, component, params)

    zall = []
    zinit = []
    for i in init_data:
        for j in i:
            zinit.append(j)

    for i in range(steps):
        zinit = np.repeat(np.expand_dims(zinit, axis=0), len(t), axis=0)

        x1, x2, x3 = deeponet.apply(params, t, zinit)
        x1t = w_t(t, zinit, 0, params)
        x1tt = w_tt(t, zinit, 0, params)
        x2t = w_t(t, zinit, 1, params)
        x2tt = w_tt(t, zinit, 1, params)
        x3t = w_t(t, zinit, 2, params)
        x3tt = w_tt(t, zinit, 2, params)

        zinit = np.array([x1[-1], x1t[-1], x1tt[-1], x2[-1], x2t[-1], x2tt[-1], x3[-1], x3t[-1], x3tt[-1]])
        zall.append(np.column_stack([x1[:-1] ,x2[:-1], x3[:-1]]))
    uall = np.concatenate(zall, axis=0)

    plt.plot(tt, uall[:,0])
    plt.plot(tt, uall[:,1])
    plt.plot(tt, uall[:,2])
    plt.grid()
    plt.legend(["u(t)", "x(t)", "y(t)"]) 
    plt.ylabel('u, x, y')
    plt.savefig(title)
    plt.clf()

    return