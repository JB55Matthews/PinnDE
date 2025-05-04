import src.pinnde as p

domain = p.domain.interval([0,1])

bound = p.boundaries.dirichletT([])

inits = p.initials.initials(domain, lambda x: 0)

data = p.data.timededata(domain, bound, inits)

test2 = p.models.pinn(data)