import jax
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

def timeStep_ordert1(steps, t_bdry, deeponet, u0, T, X, x_bdry, N_iv):
    l, n = 50, N_iv
    t = np.linspace(t_bdry[0], t_bdry[1], l)
    x = np.linspace(x_bdry[0], x_bdry[1], n)
    T, X  = np.meshgrid(t, x, indexing='ij')

    u = []
    x_points = np.linspace(x_bdry[0], x_bdry[1], N_iv)
    usensor = u0(np.expand_dims(x_points, axis=0)).numpy()
    tfinal = steps*t_bdry[1]

    for i in range(steps):
        #zinit = np.repeat(np.expand_dims(zinit, axis=0), len(t), axis=0)
        
        u_c = deeponet([np.expand_dims(T.flatten(), axis=1),
           np.expand_dims(X.flatten(), axis=1),
           usensor])[:,0]
        u_c = np.reshape(u_c, (l, n))
        #print(np.shape(u_c))

        uinit = np.repeat(usensor, repeats=l, axis=0)
        #print(np.shape(uinit))

        # For form u(t,x) = u0(x) + t/tf*v(t,x)
        u_c = uinit + T/t_bdry[1]*u_c

        # # For form u(t,x) = u0(x)*(1-t/tf) + t/tf*v(t,x)
        #u_c = uinit*(1-T/t_bdry[1]) + T/t_bdry[1]*u_c

        u.append(u_c[:-1,])

        # Next initial condition
        usensor = u_c[-1:,:]

    u = np.concatenate(u)

    t = np.linspace(t_bdry[0], steps*t_bdry[1], steps*(l-1))
    x = np.linspace(x_bdry[0], x_bdry[1], n)
    TAll, XAll  = np.meshgrid(t, x, indexing='ij')

    plt.figure()
    plt.contourf(TAll, XAll, u, 200, cmap=plt.cm.jet)
    plt.title('Neural network solution')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.colorbar()
    plt.savefig("PDE-timeStep")
    plt.clf()
    return