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
    out = nn.Dense(self.units)(b)
    return out
  

class HardConstraint(nn.Module):
  t0: float
  tfinal: float
  u0: float

  @nn.compact
  def __call__(self, inputs):
    t, nn = inputs
    return self.u0 + ((t-self.t0)/(self.tfinal-self.t0))*nn
  
class DeepONet(nn.Module):

  t0: float
  tfinal: float
  layers: int
  units: int

  @nn.compact
  def __call__(self, t, u):

    # # Add the batch dimension for branch net
    if u.ndim == 1:
      u = jnp.reshape(u, (-1,1))

    # Hard constraints
    u0 = u[:,:1]

    # Add the batch dimension for trunk net
    t = jnp.reshape(t, (-1,1))
    b = Normalize(self.t0, self.tfinal)(t)

    trunk_net = MLP(self.layers, self.units)(b)
    branch_net = MLP(self.layers, self.units)(u)

    u = CombineBranches()(trunk_net, branch_net)
    u = HardConstraint(self.t0, self.tfinal, u0)([t, u])

    # Remove the batch dimension
    u = jnp.reshape(u, (-1,))

    return u
  
# Define the collocation points over the domain [t_0, t_1]
def defineCollocationPoints(t_bdry, N, sensor_range):

    # Sample points where to evaluate the PDE
  ode_points = t_bdry[0] + (t_bdry[1] - t_bdry[0])*lhs(1, N)
  zsensors = np.random.uniform(float(sensor_range[0]), float(sensor_range[1]), size=(N, 1))

  return (ode_points, zsensors)

# DeepONet training function (standard DeepONet)
@jax.jit
def train_step(params, pdes, z):
  t = pdes[:,0]

  def u_model(t, z, params):
    return deeponet.apply(params, t, z)[0]
  def u0(t, z, params):
    return jax.vmap(u_model, [0, 0, None])(t, z, params)
  def u_t(t, z, params):
    return jax.vmap(jax.grad(u_model, 0), [0, 0, None])(t, z, params)
  def loss(params, t, z):
    ut = u_t(t, z, params)
    u = u0(t, z, params)
    parse_tree = ast.parse(equation, mode="eval")
    eqn = eval(compile(parse_tree, "<string>", "eval"))
    return jnp.mean(eqn**2)

  return jax.value_and_grad(loss)(params, t, z)

@partial(jax.jit, static_argnums=(3,))
def optimize(grads, opt_state, params, optimizer_update):
  updates, opt_state = optimizer_update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state

def train_network(params, des, zsensor, epochs):

  # Number of ICs and total number of DE collocation points
  N = des.shape[0]

  nr_batches = 1
  batch_size = len(des)//nr_batches

  lr = 1e-3
  optimizer = optax.adam(lr)
  opt_state = optimizer.init(params)

  ds_z = tf.data.Dataset.from_tensor_slices(zsensor)
  ds_de = tf.data.Dataset.from_tensor_slices(des)

  ds = tf.data.Dataset.zip((ds_de, ds_z))
  ds = ds.shuffle(N).batch(batch_size)

  epoch_loss = np.zeros(epochs)

  for i in range(epochs):

    for (des_batch, z_batch) in ds:

      z0 = z_batch[:,:1].numpy()

      loss, grads = train_step(params, des_batch.numpy(), z0)
      params, opt_state = optimize(grads, opt_state, params, optimizer.update)

      epoch_loss[i] += loss

    epoch_loss[i] /= batch_size

    if i % 100 == 0:
      print(f'Loss in epoch {i}: {epoch_loss[i]}')

  return params, epoch_loss


def startTraining(eval_points, inits, order, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn):

  global equation
  equation = eqn

  global deeponet
  deeponet = DeepONet(t[0], t[1], layers=net_layers, units=net_units)

  (de_points, zsensors) = defineCollocationPoints(t, N_sensors, sensor_range)

  params = deeponet.init(jax.random.PRNGKey(0), jnp.ones((10,)), jnp.ones((10,)))
  print(deeponet.tabulate(jax.random.PRNGKey(0), jnp.ones((10,)), jnp.ones((10,))))

  params, loss = train_network(params, de_points, zsensors, epochs)

  t = np.linspace(t[0], t[1], eval_points) 
  # Sample a new IC
  init_data1 = inits[0]+0*np.random.uniform(-1.0, 1.0, size=(1,))
  z = np.repeat(np.expand_dims(init_data1, axis=0), len(t), axis=0)

  solPred = deeponet.apply(params, t, z) 

  return loss, t, solPred, params, deeponet, de_points, zsensors
