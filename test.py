import src.pinnde as p
import numpy as np
import tensorflow as tf

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
tre = p.domain.Time_NRect(1, [0], [1], [0,1])
bound = p.boundaries.dirichlet(tre, [lambda x1: tf.sin(np.pi*x1)])
inits = p.initials.initials(tre, [lambda x1: 0+x1*0])
dat = p.data.timededata(tre, bound, inits, 100, 100, 100)
mymodel = p.models.pinn(dat, ["0.08*ux1x1 - ut"])
mymodel.train(100)