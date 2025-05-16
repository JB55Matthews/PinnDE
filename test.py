import src.pinnde as p
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Poisson
re2 = p.domain.NRect(2, [-1, -1], [1, 1])
bound2 = p.boundaries.dirichlet(re2, [lambda x1, x2: tf.cos(np.pi*x1)*tf.sin(np.pi*x2)])
dat2 = p.data.dedata(re2, bound2, 12000, 800)
mymodel = p.models.pinn(dat2, ["ux1x1 + ux2x2 - (-2*np.pi**2*tf.cos(np.pi*x1)*tf.sin(np.pi*x2))"])
epochs = 1500
mymodel.train(epochs)
p.plotters.plot_solution_prediction_2D(mymodel)
p.plotters.plot_epoch_loss(mymodel)

# Heat
# tre = p.domain.Time_NRect(1, [0], [1], [0,1])
# bound = p.boundaries.dirichlet(tre, [lambda t, x1: 0+t*0 ])
# inits = p.initials.initials(tre, [lambda x1: tf.sin(np.pi*x1)])
# dat = p.data.timededata(tre, bound, inits, 10000, 10, 100)
# mymodel = p.models.pinn(dat, ["0.08*ux1x1 - ut"])
# epochs = 300
# mymodel.train(epochs)
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
# dat = p.data.timededata(tre, bound, inits, 12000, 1000, 1000)
# mymodel = p.models.pinn(dat, ["0.08*ux1x1 + 0.08*ux2x2 - ut"])
# mymodel.train(2500)

# p.plotters.plot_solution_prediction_time2D(mymodel)
# p.plotters.plot_epoch_loss(mymodel)

# -------------
