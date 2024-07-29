from pyDOE import lhs

#will be more useful file in future when constructing adapative point selection

def defineCollocationPoints(t_bdry, N_pde):
  """
  Function which generates collocation points along t.

  Args:
    t_bdry (list): list of two elements, the interval of t to be solved on.
    N_pde (int): Number of randomly sampled collocation points along t which PINN uses in training.
  
  Returns:
    ode_points (list): randomly and evenly sampled points along t, using pyDOE's lhs function

  """

    # Sample points where to evaluate the PDE
  ode_points = t_bdry[0] + (t_bdry[1] - t_bdry[0])*lhs(1, N_pde)

  return ode_points