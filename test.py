import src.pinnde as p
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# tel = p.domain.Time_NEllipsoid(2, [0, 0], [2.5, 1.5], [0,1])
# bound = p.boundaries.dirichlet(tel, [lambda t, x1, x2: 0+t*0 ])
# inits = p.initials.initials(tel, [lambda x1, x2: tf.sin(np.pi*x1)*tf.sin(np.pi*x2)])
# dat = p.data.timepinndata(tel, bound, inits, 12000, 1000, 1000)
# mymodel = p.models.pinn(dat, ["0.08*ux1x1 + 0.08*ux2x2 - ut"])
# mymodel.train(2000)

# tr = p.domain.Time_NRect(2, [0, 0], [1, 1], [3,4])
# b =tr.sampleBoundary(10)
# print(b)
# print(np.shape(b))

# Poisson
# re2 = p.domain.NRect(2, [-1, -1], [1, 1])
# bound2 = p.boundaries.dirichlet(re2, [lambda x1, x2: tf.cos(np.pi*x1)*tf.sin(np.pi*x2)])
# dat2 = p.data.pinndata(re2, bound2, 12000, 10)
# mymodel = p.models.pinn(dat2, ["ux1x1 + ux2x2 - (-2*np.pi**2*tf.cos(np.pi*x1)*tf.sin(np.pi*x2))"])
# mymodel.train(1)
# p.plotters.plot_solution_prediction_2D(mymodel)
# p.plotters.plot_epoch_loss(mymodel)

# Linear Advection
# tre = p.domain.Time_NRect(1, [-1], [1], [0,1])
# bound = p.boundaries.periodic(tre)
# inits = p.initials.initials(tre, [lambda x1: tf.cos(np.pi*x1)])
# dat = p.data.timepinndata(tre, bound, inits, 10000, 10, 100)
# mymodel = p.models.pinn(dat, ["ut+ux1"])
# mymodel.train(300)
# p.plotters.plot_solution_prediction_time1D(mymodel)
# p.plotters.plot_epoch_loss(mymodel)

# Heat
# tre = p.domain.Time_NRect(1, [0], [1], [0,1])
# bound = p.boundaries.dirichlet(tre, [lambda t, x1: 0+t*0 ])
# inits = p.initials.initials(tre, [lambda x1: tf.sin(np.pi*x1)])
# dat = p.data.timepinndata(tre, bound, inits, 10000, 800, 600)
# mymodel = p.models.pinn(dat, ["0.08*ux1x1 - ut"])
# mymodel.train(800)
# p.plotters.plot_solution_prediction_time1D(mymodel)
# p.plotters.plot_epoch_loss(mymodel)

#klein-gordon - deepxde
# utt + auxx + by + cy^k = -xcos(t) + x^2cos(t)^2, a=-1, b=0, c=1, k=2
# Solution : xcos(t)

# tre = p.domain.Time_NRect(1, [-1], [1], [0,10])
# bound = p.boundaries.dirichlet(tre, [lambda t, x1: -(tf.cos(t)), lambda t, x1: tf.cos(t)])
# inits = p.initials.initials(tre, [lambda x1: x1, lambda x1: 0+x1*0])
# dat = p.data.timepinndata(tre, bound, inits, 15000,  1000, 500)
# mymodel = p.models.pinn(dat, ["utt - ux1x1 + u**2 - (-(x1*tf.cos(t)) + (x1**2)*((tf.cos(t))**2))"])
# epochs = 2000
# mymodel.train(epochs)
# p.plotters.plot_solution_prediction_time1D(mymodel)
# p.plotters.plot_epoch_loss(mymodel)


# Heat 3D
# tre = p.domain.Time_NRect(2, [0, 0], [1, 1], [0,1])
# tre = p.domain.Time_NEllipsoid(2, [0.5, 0.5], [0.5, 0.5], [0, 1])
# bound = p.boundaries.dirichlet(tre, [lambda t, x1, x2: 0+t*0 ])
# inits = p.initials.initials(tre, [lambda x1, x2: tf.sin(np.pi*x1)*tf.sin(np.pi*x2)])
# dat = p.data.timepinndata(tre, bound, inits, 12000, 600, 300)
# mymodel = p.models.pinn(dat, ["0.08*ux1x1 + 0.08*ux2x2 - ut"])
# mymodel.train(800)
# p.plotters.plot_solution_prediction_time2D(mymodel)
# p.plotters.plot_epoch_loss(mymodel)

#KvD
# tre = p.domain.Time_NRect(1, [-1], [1], [0,1])
# bound = p.boundaries.periodic(tre)
# inits = p.initials.initials(tre, [lambda x1: tf.cos(np.pi*x1)])
# dat = p.data.timepinndata(tre, bound, inits, 12000, 10, 1000)
# mymodel = p.models.pinn(dat, ["ut+u*ux1-(-0.0025)*ux1x1x1"])
# mymodel.train(500)
# p.plotters.plot_epoch_loss(mymodel)
# p.plotters.plot_solution_prediction_time1D(mymodel)

#Burgers
# tre = p.domain.Time_NRect(1, [-1], [1], [0,1])
# bound = p.boundaries.dirichlet(tre, [lambda t, x1: 0+t*0 ])
# inits = p.initials.initials(tre, [lambda x1: -tf.sin(np.pi*x1)])
# dat = p.data.timepinndata(tre, bound, inits, 10000, 600, 400)
# mymodel = p.models.pinn(dat, ["ut+u*ux1-(0.01/np.pi)*ux1x1"])
# mymodel.train(500)
# p.plotters.plot_solution_prediction_time1D(mymodel)
# p.plotters.plot_epoch_loss(mymodel)


# test ode
# tre = p.domain.NRect(1, [0], [1])
# cond = p.boundaries.odeicbc(tre, [[0.5, 1], [2]], "ic")
# # cond = p.boundaries.odeicbc(tre, [[0.5, 1.11, 1, 0.12], [2, 1.12]], "bc")
# dat = p.data.pinndata(tre, cond, 1000, 100)
# mymodel = p.models.pinn(dat, ["u1x1x1 + u1", "u2x1+u1"])
# mymodel.train(500)
# p.plotters.plot_epoch_loss(mymodel)
# p.plotters.plot_solution_prediction_1D(mymodel)

# test system - advec and burgers
# tre = p.domain.Time_NRect(1, [-1], [1], [0, 1])
# bound = p.boundaries.periodic(tre)
# inits = p.initials.initials(tre, [[lambda x1: tf.cos(np.pi*x1)], [lambda x1: tf.cos(np.pi*x1)]])
# dat = p.data.timepinndata(tre, bound, inits, 12000, 10, 500)
# mymodel = p.models.pinn(dat, ["u1t+u1x1", "u2t+u2*u2x1-(-0.0025)*u2x1x1x1"])
# mymodel.train(1500)
# p.plotters.plot_epoch_loss(mymodel)
# p.plotters.plot_solution_prediction_time1D(mymodel)


# tre = p.domain.Time_NRect(1, [-1], [1], [0, 1])
# bound = p.boundaries.dirichlet(tre, [lambda t, x1: 0+t*0])
# inits = p.initials.initials(tre, [[lambda x1: tf.sin((np.pi/2)*x1 + (np.pi/2))], [lambda x1: -tf.sin(np.pi*x1)]])
# dat = p.data.timepinndata(tre, bound, inits, 12000, 1000, 1000)
# mymodel = p.models.pinn(dat, ["0.08*u1x1x1 - u1t", "u2t+u2*u2x1-(0.01/np.pi)*u2x1x1"])
# mymodel.train(1500)
# p.plotters.plot_epoch_loss(mymodel)
# p.plotters.plot_solution_prediction_time1D(mymodel)


# DeepONet --------------

# Heat 1+1
# tre = p.domain.Time_NRect(1, [0], [1], [0,1])
# bound = p.boundaries.dirichlet(tre, [lambda t, x1: 0+t*0 ])
# inits = p.initials.initials(tre, [lambda x1: tf.sin(np.pi*x1)])
# dat = p.data.timedondata(tre, bound, inits, 12000, 1000, 1000, 1000)
# mymodel = p.models.deeponet(dat, ["0.08*ux1x1 - ut"])
# mymodel.train(1500)
# p.plotters.plot_solution_prediction_time1D(mymodel)
# p.plotters.plot_epoch_loss(mymodel)

# Heat 3D
# # tre = p.domain.Time_NRect(2, [0, 0], [1, 1], [0,1])
# tre = p.domain.Time_NEllipsoid(2, [0.5, 0.5], [0.5, 0.5], [0, 1])
# bound = p.boundaries.dirichlet(tre, [lambda t, x1, x2: 0+t*0 ])
# inits = p.initials.initials(tre, [lambda x1, x2: tf.sin(np.pi*x1)*tf.sin(np.pi*x2)])
# dat = p.data.timedondata(tre, bound, inits, 12000, 600, 600, 2000)
# mymodel = p.models.deeponet(dat, ["0.08*ux1x1 + 0.08*ux2x2 - ut"])
# # dat = p.data.timepinndata(tre, bound, inits, 12000, 600, 600)
# # mymodel = p.models.pinn(dat, ["0.08*ux1x1 + 0.08*ux2x2 - ut"])
# mymodel.train(1200)
# p.plotters.plot_solution_prediction_time2D(mymodel)
# p.plotters.plot_epoch_loss(mymodel)

# Poisson
re2 = p.domain.NRect(2, [-1, -1], [1, 1])
bound2 = p.boundaries.dirichlet(re2, [lambda x1, x2: tf.cos(np.pi*x1)*tf.sin(np.pi*x2)])
dat2 = p.data.dondata(re2, bound2, 12000, 1000, 2000)
mymodel = p.models.deeponet(dat2, ["ux1x1 + ux2x2 - (-2*np.pi**2*tf.cos(np.pi*x1)*tf.sin(np.pi*x2))"])
mymodel.train(1500)
p.plotters.plot_solution_prediction_2D(mymodel)
p.plotters.plot_epoch_loss(mymodel)

# Linear Advection
# tre = p.domain.Time_NRect(1, [-1], [1], [0,1])
# bound = p.boundaries.periodic(tre)
# inits = p.initials.initials(tre, [lambda x1: tf.cos(np.pi*x1)])
# dat = p.data.timedondata(tre, bound, inits, 12000, 1000, 1000, 1000)
# mymodel = p.models.deeponet(dat, ["ut+ux1"])
# # dat = p.data.timepinndata(tre, bound, inits, 12000, 4, 600)
# # mymodel = p.models.pinn(dat, ["ut+ux1"])
# mymodel.train(1200)
# p.plotters.plot_solution_prediction_time1D(mymodel)
# p.plotters.plot_epoch_loss(mymodel)

# -------------
