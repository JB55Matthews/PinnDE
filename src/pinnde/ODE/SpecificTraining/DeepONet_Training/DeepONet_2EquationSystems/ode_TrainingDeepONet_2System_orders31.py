import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from functools import partial
import tensorflow as tf
from pyDOE import lhs
import numpy as np
import ast

from jax import config
config.update("jax_enable_x64", True)

# Define the normalization layer
class Normalize(nn.Module):
  xmin: float
  xmax: float

  @nn.compact
  def __call__(self, x):
    return 2.0*(x-self.xmin)/(self.xmax - self.xmin) - 1.0
  
class CombineBranches(nn.Module):
  @nn.compact
  def __call__(self, inp1, inp2):
    mult = jnp.sum(inp1*inp2, axis=1)
    # For consistency, we add the batch dimension back here
    out = jnp.reshape(mult, (-1,1))
    return out
  
class MLP(nn.Module):

  layers: int
  units: int

  @nn.compact
  def __call__(self, inp):
    b = inp
    for i in range(self.layers-1):
      b = nn.Dense(self.units)(b)
      b = nn.tanh(b)
    out = nn.Dense(4*self.units)(b)
    return out
  

class HardConstraint(nn.Module):
  t0: float
  tfinal: float
  u0: float

  @nn.compact
  def __call__(self, inputs):
    t, nn = inputs
    return self.u0 + ((t-self.t0)/(self.tfinal-self.t0))*nn
  
class HardConstraint3(nn.Module):
  t0: float
  tfinal: float
  u0: float
  ut0: float
  utt0: float

  @nn.compact
  def __call__(self, inputs):
    t, nn = inputs
    return self.u0 + (t-self.t0)*self.ut0 + (t-self.t0)**2*self.utt0 + ((t-self.t0)/(self.tfinal-self.t0))**3*nn

class DeepONet(nn.Module):

  t0: float
  tfinal: float
  layers: int
  units: int

  @nn.compact
  def __call__(self, t, u):

    # Add the batch dimension
    if u.ndim == 1:
      u = jnp.reshape(u, (1,-1))

    x1_0 = u[:,:1]
    x1t_0 = u[:,1:2]
    x1tt_0 = u[:,2:3]
    x2_0 = u[:,3:]

    # Add the batch dimension
    t = jnp.reshape(t, (-1,1))

    b = Normalize(self.t0, self.tfinal)(t)

    trunk_net = MLP(self.layers, self.units)(b)
    branch_net = MLP(self.layers, self.units)(u)

    # Split into trunk/branch for q and p
    trunk_x1 = trunk_net[:,:3*self.units]
    trunk_x2 = trunk_net[:,3*self.units:]

    branch_x1 = branch_net[:,:3*self.units]
    branch_x2 = branch_net[:,3*self.units:]

    x1 = CombineBranches()(trunk_x1, branch_x1)
    x2 = CombineBranches()(trunk_x2, branch_x2)

    x1 = HardConstraint3(self.t0, self.tfinal, x1_0, x1t_0, x1tt_0)([t, x1])
    x2 = HardConstraint(self.t0, self.tfinal, x2_0)([t, x2])

    # Remove the batch dimension
    x1 = jnp.reshape(x1, (-1,))
    x2 = jnp.reshape(x2, (-1,))

    return x1, x2
  
# Define the collocation points over the domain [t_0, t_1]
def defineCollocationPoints(t_bdry, N, sensor_range):

    # Sample points where to evaluate the PDE
  ode_points = t_bdry[0] + (t_bdry[1] - t_bdry[0])*lhs(1, N)
  zsensors = np.random.uniform(float(sensor_range[0]), float(sensor_range[1]), size=(N, 4))

  return (ode_points, zsensors)

# DeepONet training function (standard DeepONet)
#@jax.jit
@partial(jax.jit, static_argnames=['orders_case'])
def train_step(params, pdes, z, orders_case):
  t = pdes[:,0]


  def w_model(t, z, component, params):
    return deeponet.apply(params, t, z)[component][0]
  def w0(t, z, component, params):
    return jax.vmap(w_model, [0, 0, None, None])(t, z, component, params)
  def w_t(t, z, component, params):
    return jax.vmap(jax.grad(w_model, 0), [0, 0, None, None])(t, z, component, params)
  def w_tt(t, z, component, params):
    return jax.vmap(jax.grad(jax.grad(w_model, 0),0), [0, 0, None, None])(t, z, component, params)
  def w_ttt(t, z, component, params):
    return jax.vmap(jax.grad(jax.grad(jax.grad(w_model, 0),0),0), [0, 0, None, None])(t, z, component, params)
  def loss(params, t, z, orders_case):
    if (orders_case == 1):
      u = w0(t, z, int(eqn1[-1]), params)
      x = w0(t, z, int(eqn2[-1]), params)
      ut = w_t(t, z, int(eqn1[-1]), params)
      xt = w_t(t, z, int(eqn2[-1]), params)
      utt = w_tt(t, z, int(eqn1[-1]), params)
      uttt = w_ttt(t, z, int(eqn1[-1]), params)
    elif (orders_case == 2):
      u = w0(t, z, int(eqn2[-1]), params)
      x = w0(t, z, int(eqn1[-1]), params)
      ut = w_t(t, z, int(eqn2[-1]), params)
      xt = w_t(t, z, int(eqn1[-1]), params)
      xtt = w_tt(t, z, int(eqn1[-1]), params)
      xttt = w_ttt(t, z, int(eqn1[-1]), params)
    
    parse_tree1 = ast.parse(eqn1[:-1], mode="eval")
    e1 = eval(compile(parse_tree1, "<string>", "eval"))
    parse_tree2 = ast.parse(eqn2[:-1], mode="eval")
    e2 = eval(compile(parse_tree2, "<string>", "eval"))

    return jnp.mean(e1**2 + e2**2)

  return jax.value_and_grad(loss)(params, t, z, orders_case)

@partial(jax.jit, static_argnums=(3,))
def optimize(grads, opt_state, params, optimizer_update):
  updates, opt_state = optimizer_update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state

def train_network(params, des, zsensor, epochs, orders_case):

  # Number of ICs and total number of DE collocation points
  N = des.shape[0]

  nr_batches = 1
  batch_size = len(des)//nr_batches

  # Learning rate schedule
  lr = 1e-3
  # schedule = optax.linear_schedule(lr, lr//10, epochs)
  optimizer = optax.adam(lr)

  opt_state = optimizer.init(params)

  ds_z = tf.data.Dataset.from_tensor_slices(zsensor)
  ds_de = tf.data.Dataset.from_tensor_slices(des)

  ds = tf.data.Dataset.zip((ds_de, ds_z))
  ds = ds.shuffle(N).batch(batch_size)

  epoch_loss = np.zeros(epochs)

  for i in range(epochs):

    for (des_batch, z_batch) in ds:

      loss, grads = train_step(params, des_batch.numpy(), z_batch.numpy(), orders_case)
      params, opt_state = optimize(grads, opt_state, params, optimizer.update)

      epoch_loss[i] += loss

    epoch_loss[i] /= batch_size

    if i % 100 == 0:
      print(f'Loss in epoch {i}: {epoch_loss[i]}')

  return params, epoch_loss


def startTraining(eval_points, inits, orig_orders, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqns):
  
  global eqn1, eqn2
  eqn1 = eqns[0]
  eqn1 = eqn1 + "0"
  eqn2 = eqns[1]
  eqn2 = eqn2 + "1"
  global deeponet
  deeponet = DeepONet(t[0], t[1], layers=net_layers, units=net_units)

  if ((orig_orders[0] == 3) and (orig_orders[1] == 1)):
    orders_case = 1
  elif ((orig_orders[0] == 1) and (orig_orders[1] == 3)):
    orders_case = 2


  (de_points, zsensors) = defineCollocationPoints(t, N_sensors, sensor_range)

  params = deeponet.init(jax.random.PRNGKey(0), jnp.ones((10,)), jnp.ones((10,4)))

  params, loss = train_network(params, de_points, zsensors, epochs, orders_case)

  t = np.linspace(t[0], t[1], eval_points) 
  init_data = []
  for i in inits:
    for j in i:
      init_data.append(j)
  z = np.repeat(np.expand_dims(init_data, axis=0), len(t), axis=0)

  solPred = deeponet.apply(params, t, z) 

  return loss, t, solPred, params, deeponet, de_points, zsensors
