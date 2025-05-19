import src.pinnde as p
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# tel = p.domain.Time_NEllipsoid(2, [0, 0], [2.5, 1.5], [0,1])
# bound = p.boundaries.dirichlet(tel, [lambda t, x1, x2: 0+t*0 ])
# inits = p.initials.initials(tel, [lambda x1, x2: tf.sin(np.pi*x1)*tf.sin(np.pi*x2)])
# dat = p.data.timededata(tel, bound, inits, 12000, 1000, 1000)
# mymodel = p.models.pinn(dat, ["0.08*ux1x1 + 0.08*ux2x2 - ut"])
# mymodel.train(2000)

# tr = p.domain.Time_NRect(2, [0, 0], [1, 1], [3,4])
# b =tr.sampleBoundary(10)
# print(b)
# print(np.shape(b))

# Poisson
# re2 = p.domain.NRect(2, [-1, -1], [1, 1])
# bound2 = p.boundaries.dirichlet(re2, [lambda x1, x2: tf.cos(np.pi*x1)*tf.sin(np.pi*x2)])
# dat2 = p.data.dedata(re2, bound2, 12000, 800)
# mymodel = p.models.pinn(dat2, ["ux1x1 + ux2x2 - (-2*np.pi**2*tf.cos(np.pi*x1)*tf.sin(np.pi*x2))"])
# epochs = 1500
# mymodel.train(epochs)
# p.plotters.plot_solution_prediction_2D(mymodel)
# p.plotters.plot_epoch_loss(mymodel)

# Linear Advection
tre = p.domain.Time_NRect(1, [-1], [1], [0,1])
bound = p.boundaries.periodic(tre)
inits = p.initials.initials(tre, [lambda x: tf.cos(np.pi*x)])
dat = p.data.timededata(tre, bound, inits, 10000, 100, 400)
mymodel = p.models.pinn(dat, ["ut+ux1"])
mymodel.train(400)
p.plotters.plot_solution_prediction_time1D(mymodel)
p.plotters.plot_epoch_loss(mymodel)

# Heat
# tre = p.domain.Time_NRect(1, [0], [1], [0,1])
# bound = p.boundaries.dirichlet(tre, [lambda t, x1: 0+t*0 ])
# inits = p.initials.initials(tre, [lambda x1: tf.sin(np.pi*x1)])
# dat = p.data.timededata(tre, bound, inits, 10000, 10, 10)
# mymodel = p.models.pinn(dat, ["0.08*ux1x1 - ut"])
# mymodel.train(1)
# p.plotters.plot_solution_prediction_time1D(mymodel)
# p.plotters.plot_epoch_loss(mymodel)

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
# p.plotters.plot_solution_prediction_time1D(mymodel)
# p.plotters.plot_epoch_loss(mymodel)


# Heat 3D

# tre = p.domain.Time_NRect(2, [0, 0], [1, 1], [0,1])
# bound = p.boundaries.dirichlet(tre, [lambda t, x1, x2: 0+t*0 ])
# inits = p.initials.initials(tre, [lambda x1, x2: tf.sin(np.pi*x1)*tf.sin(np.pi*x2)])
# dat = p.data.timededata(tre, bound, inits, 12000, 10, 10)
# mymodel = p.models.pinn(dat, ["0.08*ux1x1 + 0.08*ux2x2 - ut"])
# mymodel.train(1)

# p.plotters.plot_solution_prediction_time2D(mymodel)
# p.plotters.plot_epoch_loss(mymodel)


# test
# tre = p.domain.NRect(4, [-1, -3, -2, -4], [1, 3, 2, 4])
# bound = p.boundaries.neumann(tre, [lambda t, x1, x2, x3: 4+t*0, lambda t, x1, x2, x3: 5+t*0, lambda t, x1, x2, x3: 6+t*0, lambda t, x1, x2, x3: 7+t*0, lambda t, x1, x2, x3: 8+t*0, 
#                                      lambda t, x1, x2, x3: 9+t*0, lambda t, x1, x2, x3: 10+t*0, lambda t, x1, x2, x3: 11+t*0])
# # inits = p.initials.initials(tre, [lambda x1, x2, x3, x4: tf.sin(np.pi*x1)*tf.sin(np.pi*x2)])
# dat = p.data.dedata(tre, bound, 12000, 16)
# mymodel = p.models.pinn(dat, ["0.08*ux1x1 + 0.08*ux2x2"])
# mymodel.train(1)

# -------------
