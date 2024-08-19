import numpy as np
from pyDOE import lhs

def defineCollocationPoints_tx(t_bdry, x_bdry, initial_t, t_order, N_pde, N_iv):
  """
  Function which generates collocation points along t and x, and which generates points for learning inital conditions.

  Args:
    t_bdry (list): list of two elements, the interval of t to be solved on.
    x_bdry (list): list of two elements, the interval of x to be solved on.
    initial_t (lambda): Inital function for t=t0, as a python lambda funciton, with t0 being inital t in t_bdry.
    t_order (int): Order of t in equation (highest derivative of t used).
    N_pde (int): Number of randomly sampled collocation points along t and x which PINN uses in training.
    N_iv (int): Number of randomly sampled collocation points along inital t which PINN uses in training.
  
  Returns:
    pde_points (np.column_stack): Randomly and evenly sampled points along t and x, using pyDOE's lhs function
    inits (np.column_stack): Stack of list containing just t0 value (t_init), list of random and evenly sampled points 
      along t0 (x_init) using lhs, and lists containing inital value functions at x_init for each function.

  """

  if t_order == 1:
    u0 = initial_t[0]
  elif t_order == 2:
    u0, ut0 = initial_t[0], initial_t[1]
  elif t_order == 3:
    u0, ut0 = initial_t[0], initial_t[1]
    utt0 = initial_t[2]

  tx_min = np.array([t_bdry[0], x_bdry[0]])
  tx_max = np.array([t_bdry[1], x_bdry[1]])  
  pde_points = tx_min + (tx_max - tx_min)*lhs(2, N_pde)
  t_pde = pde_points[:,0]
  x_pde = pde_points[:,1]

  pdes = np.column_stack([t_pde, x_pde]).astype(np.float64)

  # Sample points where to evaluate the initial values
  init_points = tx_min[1:] + (tx_max[1:] - tx_min[1:])*lhs(1, N_iv)
  x_init = init_points
  t_init = t_bdry[0]+ 0.0*x_init
  if t_order == 1:
    u_init = u0(x_init)
    inits = np.column_stack([t_init, x_init, u_init]).astype(np.float64)
  elif t_order == 2:
    u_init = u0(x_init)
    ut_init = ut0(x_init)
    inits = np.column_stack([t_init, x_init, u_init, ut_init]).astype(np.float64)
  elif t_order == 3:
    u_init = u0(x_init)
    ut_init = ut0(x_init)
    utt_init = utt0(x_init)
    inits = np.column_stack([t_init, x_init, u_init, ut_init, utt_init]).astype(np.float64)

  return pdes, inits


def defineCollocationPoints_2var(fst_bdry, sec_bdry, N_pde):
  """
  Function which generates collocation points along t and x or x and y.

  Args:
    fst_bdry (list): list of two elements, the interval of t/x to be solved on.
    sec_bdry (list): list of two elements, the interval of x/y to be solved on.
    N_pde (int): Number of randomly sampled collocation points along t/x and x/y which PINN uses in training.
  
  Returns:
    pde_points (np.column_stack): Randomly and evenly sampled points along t/x and x/y, using pyDOE's lhs function

  """

  p_min = np.array([fst_bdry[0], sec_bdry[0]])
  p_max = np.array([fst_bdry[1], sec_bdry[1]])
  pde_points = p_min + (p_max - p_min)*lhs(2, N_pde)
  fst_pde = pde_points[:,0]
  sec_pde = pde_points[:,1]
  pdes = np.column_stack([fst_pde, sec_pde]).astype(np.float64)
  return pdes

# def defineCollocationPoints_DON_tx(N_sensors ,N_iv, t_bdry, x_bdry):

#     # Uniform random sampling for PDE points
#     tx_min = np.array([t_bdry[0], x_bdry[0]])
#     tx_max = np.array([t_bdry[1], x_bdry[1]])
#     pde_points = tx_min + (tx_max - tx_min)*lhs(2, N_sensors)
#     t_pde = pde_points[:, 0]
#     x_pde = pde_points[:, 1]

#     usensors = np.random.uniform(float(min(tx_min)), float(max(tx_max)), size=(N_sensors, N_iv))
#     pdes = np.column_stack([t_pde, x_pde]).astype(np.float32)

#     return (pdes, usensors)

def defineCollocationPoints_DON_tx(t_bdry, x_bdry, initial_t, t_order, N_pde, N_iv, N_sensors, sensor_range):
  """
  Function which generates collocation points along t and x,which generates points for learning inital conditions, 
  and which generates sensors for DeepONet.

  Args:
    t_bdry (list): list of two elements, the interval of t to be solved on.
    x_bdry (list): list of two elements, the interval of x to be solved on.
    initial_t (lambda): Inital function for t=t0, as a python lambda funciton, with t0 being inital t in t_bdry.
    t_order (int): Order of t in equation (highest derivative of t used).
    N_pde (int): Number of randomly sampled collocation points along t and x which PINN uses in training.
    N_iv (int): Number of randomly sampled collocation points along inital t which PINN uses in training.
    N_sensors (int): Number of sensors in which network learns over. 
    sensor_range (list): Range in which sensors are sampled over.
  
  Returns:
    pde_points (np.column_stack): Randomly and evenly sampled points along t and x, using pyDOE's lhs function
    inits (np.column_stack): Stack of list containing just t0 value (t_init), list of random and evenly sampled points 
      along t0 (x_init) using lhs, and lists containing inital value functions at x_init for each function.
    usensors (list): Uniformly sampled sensor points in sensor range in shape (N_sensors, N_iv)

  """

  if t_order == 1:
    u0 = initial_t[0]
  elif t_order == 2:
    u0, ut0 = initial_t[0], initial_t[1]
  elif t_order == 3:
    u0, ut0 = initial_t[0], initial_t[1]
    utt0 = initial_t[2]

  tx_min = np.array([t_bdry[0], x_bdry[0]])
  tx_max = np.array([t_bdry[1], x_bdry[1]])  
  pde_points = tx_min + (tx_max - tx_min)*lhs(2, N_pde)
  t_pde = pde_points[:,0]
  x_pde = pde_points[:,1]

  pdes = np.column_stack([t_pde, x_pde]).astype(np.float64)

  # Sample points where to evaluate the initial values
  init_points = tx_min[1:] + (tx_max[1:] - tx_min[1:])*lhs(1, N_iv)
  x_init = init_points
  t_init = t_bdry[0]+ 0.0*x_init
  if t_order == 1:
    u_init = u0(x_init)
    inits = np.column_stack([t_init, x_init, u_init]).astype(np.float64)
  elif t_order == 2:
    u_init = u0(x_init)
    ut_init = ut0(x_init)
    inits = np.column_stack([t_init, x_init, u_init, ut_init]).astype(np.float64)
  elif t_order == 3:
    u_init = u0(x_init)
    ut_init = ut0(x_init)
    utt_init = utt0(x_init)
    inits = np.column_stack([t_init, x_init, u_init, ut_init, utt_init]).astype(np.float64)

  usensors = np.random.uniform(float(sensor_range[0]), float(sensor_range[1]), size=(N_sensors, N_iv))


  return pdes, inits, usensors

def defineCollocationPoints_DON_2var(fst_bdry, sec_bdry, N_pde, N_bc, N_sensors, sensor_range):
  """
  Function which generates collocation points along t and x or x and y, and which generates sensors for DeepONet.

  Args:
    fst_bdry (list): list of two elements, the interval of t/x to be solved on.
    sec_bdry (list): list of two elements, the interval of x/y to be solved on.
    N_pde (int): Number of randomly sampled collocation points along t/x and x/y which PINN uses in training.
    N_bc (int): Number of randomly sampled collocation points along boundaries which PINN uses in training.
    N_sensors (int): Number of sensors in which network learns over. 
    sensor_range (list): Range in which sensors are sampled over.
  
  Returns:
    pde_points (np.column_stack): Randomly and evenly sampled points along t/x and x/y, using pyDOE's lhs function
    usensors (list): Uniformly sampled sensor points in sensor range in shape (N_sensors, N_bc)

  """

  p_min = np.array([fst_bdry[0], sec_bdry[0]])
  p_max = np.array([fst_bdry[1], sec_bdry[1]])
  pde_points = p_min + (p_max - p_min)*lhs(2, N_pde)
  fst_pde = pde_points[:,0]
  sec_pde = pde_points[:,1]
  pdes = np.column_stack([fst_pde, sec_pde]).astype(np.float64)

  usensors = np.random.uniform(float(sensor_range[0]), float(sensor_range[1]), size=(N_sensors, N_bc)).astype(np.float64)
  return pdes, usensors
  

def defineCollocationPoints_txy(t_bdry, x_bdry, y_bdry, initial_t, t_order, N_pde, N_iv):

  if t_order == 1:
    u0 = initial_t[0]
  elif t_order == 2:
    u0, ut0 = initial_t[0], initial_t[1]
  elif t_order == 3:
    u0, ut0 = initial_t[0], initial_t[1]
    utt0 = initial_t[2]

  txy_min = np.array([t_bdry[0], x_bdry[0], y_bdry[0]])
  txy_max = np.array([t_bdry[1], x_bdry[1], y_bdry[1]])  
  pde_points = txy_min + (txy_max - txy_min)*lhs(3, N_pde)
  t_pde = pde_points[:,0]
  x_pde = pde_points[:,1]
  y_pde = pde_points[:,2]

  pdes = np.column_stack([t_pde, x_pde, y_pde]).astype(np.float64)

  # Sample points where to evaluate the initial values
  init_points = txy_min[1:] + (txy_max[1:] - txy_min[1:])*lhs(2, N_iv)
  x_init = init_points[:,0]
  y_init = init_points[:,1]
  t_init = t_bdry[0]+ 0.0*x_init
  if t_order == 1:
    u_init = u0(x_init, y_init)
    inits = np.column_stack([t_init, x_init, y_init, u_init]).astype(np.float64)
  elif t_order == 2:
    u_init = u0(x_init, y_init)
    ut_init = ut0(x_init, y_init)
    inits = np.column_stack([t_init, x_init, y_init, u_init, ut_init]).astype(np.float64)
  elif t_order == 3:
    u_init = u0(x_init, y_init)
    ut_init = ut0(x_init, y_init)
    utt_init = utt0(x_init, y_init)
    inits = np.column_stack([t_init, x_init, y_init, u_init, ut_init, utt_init]).astype(np.float64)

  return pdes, inits
