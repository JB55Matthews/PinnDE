import src.pinnde as p
import numpy as np

tre = p.domain.Time_NRect(2, [-1, -1], [1, 1], [0,2])
bound = p.boundaries.dirichlet(tre, [lambda x, y: 0+x*0+y*0])
inits = p.initials.initials(tre, [lambda x, y: (np.sin(np.pi*x)*np.sin(np.pi*y))/2])
dat = p.data.timededata(tre, bound, inits, 100, 100, 100)
mymodel = p.models.pinn(dat, ["ux + uy"])
mymodel.train(100)

# re2 = p.domain.NRect(4, [-1, 0, 10, -5], [1, 1, 12, -6])
# bound2 = p.boundaries.dirichlet(re2, [lambda x1, x2, x3, x4: 0+x1*0])
# dat2 = p.data.dedata(re2, bound2, 100, 100)
# mymodel2 = p.models.pinn(dat2)