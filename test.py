import src.pinnde as p
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# tre = p.domain.Time_NRect(2, [-1, -1], [1, 1], [0,2])
# bound = p.boundaries.dirichlet(tre, [lambda x1, x2: 0+x1*0+x2*0])
# inits = p.initials.initials(tre, [lambda x1, x2: (np.sin(np.pi*x1)*np.sin(np.pi*x2))/2])
# dat = p.data.timededata(tre, bound, inits, 10, 10, 10)
# mymodel = p.models.pinn(dat, ["0.08*ux1x1 + 0.08*ux2x2 - ut"])
# mymodel.train(1)

# re2 = p.domain.NRect(4, [-1, 0, 10, -5], [1, 1, 12, -6])
# bound2 = p.boundaries.dirichlet(re2, [lambda x1, x2, x3, x4: 0+x1*0])
# dat2 = p.data.dedata(re2, bound2, 100, 100)
# mymodel2 = p.models.pinn(dat2)

# Heat
# tre = p.domain.Time_NRect(1, [0], [1], [0,1])
# bound = p.boundaries.dirichlet(tre, [lambda t, x1: 0+t*0 ])
# inits = p.initials.initials(tre, [lambda x1: tf.sin(np.pi*x1)])
# dat = p.data.timededata(tre, bound, inits, 10000, 400, 100)
# mymodel = p.models.pinn(dat, ["0.08*ux1x1 - ut"])
# epochs = 300
# mymodel.train(epochs)

#klein-gordon - deepxde
# utt + auxx + by + cy^k = -xcos(t) + x^2cos(t)^2, a=-1, b=0, c=1, k=2
# Solution : xcos(t)

# tre = p.domain.Time_NRect(1, [-1], [1], [0,10])
# bound = p.boundaries.dirichlet(tre, [lambda t, x1: -(tf.cos(t)), lambda t, x1: tf.cos(t)])
# inits = p.initials.initials(tre, [lambda x1: x1, lambda x1: 0+x1*0])
# dat = p.data.timededata(tre, bound, inits, 15000,  1000, 500)
# mymodel = p.models.pinn(dat, ["utt - ux1x1 + u**2 - (-(x1*tf.cos(t)) + (x1**2)*((tf.cos(t))**2))"])
# epochs = 2000
# mymodel.train(epochs)

# network = mymodel.get_network()
# T, X = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200), indexing='ij')
# sols = network([np.expand_dims(T.flatten(), axis=1), np.expand_dims(X.flatten(), axis=1)])
# sols = np.reshape(sols, (200, 200))
# plt.figure()
# plt.contourf(T, X, sols, 200, cmap=plt.cm.jet)
# plt.title('Neural network solution')
# plt.xlabel('t')
# plt.ylabel('x')
# plt.colorbar()
# plt.savefig("PDE-sol-pred")
# plt.clf()
# plt.figure()
# plt.semilogy(np.linspace(1, epochs, epochs),mymodel.get_epoch_loss())
# plt.grid()
# plt.xlabel('epochs')
# plt.ylabel('Epoch loss')
# plt.savefig("PDE-epoch-loss")
# plt.clf()


# Heat 3D
tre = p.domain.Time_NRect(2, [0, 0], [1, 1], [0,1])
bound = p.boundaries.dirichlet(tre, [lambda t, x1, x2: 0+t*0 ])
inits = p.initials.initials(tre, [lambda x1, x2: tf.sin(np.pi*x1)*tf.sin(np.pi*x2)])
dat = p.data.timededata(tre, bound, inits, 12000, 1000, 1000)
mymodel = p.models.pinn(dat, ["0.08*ux1x1 + 0.08*ux2x2 - ut"])
epochs = 2500
mymodel.train(epochs)

network = mymodel.get_network()
X1, X2 = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200), indexing='ij')
T0 = np.linspace(0, 0, 200*200)
T0 = T0.reshape((200*200, 1))
T05 = np.linspace(0.5, 0.5, 200*200)
T05 = T05.reshape((200*200, 1))
T1 = np.linspace(1, 1, 200*200)
T1 = T1.reshape((200*200, 1))

sols0 = network([T0, np.expand_dims(X1.flatten(), axis=1), np.expand_dims(X2.flatten(), axis=1)])
sols0 = np.reshape(sols0, (200, 200))
sols05 = network([T05, np.expand_dims(X1.flatten(), axis=1), np.expand_dims(X2.flatten(), axis=1)])
sols05 = np.reshape(sols05, (200, 200))
sols1 = network([T1, np.expand_dims(X1.flatten(), axis=1), np.expand_dims(X2.flatten(), axis=1)])
sols1 = np.reshape(sols1, (200, 200))

plt.figure()
plt.contourf(X1, X2, sols0, 200, cmap=plt.cm.jet)
plt.title('Neural network solution - Time 0')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar()
plt.savefig("PDE-sol-pred-Heat3D-T0")
plt.clf()

plt.figure()
plt.contourf(X1, X2, sols05, 200, cmap=plt.cm.jet)
plt.title('Neural network solution - Time 0.5')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar()
plt.savefig("PDE-sol-pred-Heat3D-T05")
plt.clf()

plt.figure()
plt.contourf(X1, X2, sols1, 200, cmap=plt.cm.jet)
plt.title('Neural network solution - Time 1')
plt.xlabel('t')
plt.ylabel('x')
plt.colorbar()
plt.savefig("PDE-sol-pred-Heat3D-T1")
plt.clf()

plt.figure()
plt.semilogy(np.linspace(1, epochs, epochs),mymodel.get_epoch_loss())
plt.grid()
plt.xlabel('epochs')
plt.ylabel('Epoch loss')
plt.savefig("PDE-epoch-loss-Heat3D")
plt.clf()
