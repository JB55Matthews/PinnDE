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
    out = nn.Dense(2*self.units)(b)
    return out

  
class DeepONet(nn.Module):

  t0: float
  tfinal: float
  layers: int
  units: int

  @nn.compact
  def __call__(self, t, u, ut):

    # # Add the batch dimension for branch net
    if u.ndim == 1:
      u = jnp.reshape(u, (-1,1))
    if ut.ndim == 1:
      ut = jnp.reshape(ut, (-1,1))

    # Combine for branch net
    z = jnp.concatenate([u, ut], axis=1)

    # Add the batch dimension for trunk net
    t = jnp.reshape(t, (-1,1))
    b = Normalize(self.t0, self.tfinal)(t)

    trunk_net = MLP(self.layers, self.units)(b)
    branch_net = MLP(self.layers, self.units)(z)

    u = CombineBranches()(trunk_net, branch_net)

    # Remove the batch dimension
    u = jnp.reshape(u, (-1,))
  
    return u
  
# Define the collocation points over the domain [t_0, t_1]
def defineCollocationPoints(t_bdry, N, sensor_range):

    # Sample points where to evaluate the PDE
  ode_points = t_bdry[0] + (t_bdry[1] - t_bdry[0])*lhs(1, N)
  zsensors = np.random.uniform(float(sensor_range[0]), float(sensor_range[1]), size=(N, 2))

  return (ode_points, zsensors)

# DeepONet training function (standard DeepONet)
@jax.jit
def train_step(params, pdes, z, zt):
  t = pdes[:,0]

  def u_model(t, z, zt, params):
    return deeponet.apply(params, t, z, zt)[0]
  def u0(t, z, zt, params):
    return jax.vmap(u_model, [0, 0, 0, None])(t, z, zt, params)
  def u_t(t, z, zt, params):
    return jax.vmap(jax.grad(u_model, 0), [0, 0, 0, None])(t, z, zt, params)
  def u_tt(t, z, zt, params):
    return jax.vmap(jax.grad(jax.grad(u_model,0), 0), [0, 0, 0, None])(t, z, zt, params)
  def loss(params, t, z, zt):
    utt = u_tt(t, z, zt, params)
    ut = u_t(t, z, zt, params)
    u = u0(t, z, zt, params)
    parse_tree = ast.parse(equation, mode="eval")
    eqn = eval(compile(parse_tree, "<string>", "eval"))

    x0 = np.repeat(np.expand_dims(t_0, axis=0), len(t), axis=0)
    u0_pred = u0(x0, z, zt, params)
    eqn2 = init_0 - u0_pred

    xt0 = np.repeat(np.expand_dims(t_0, axis=0), len(t), axis=0)
    ut0_pred = u_t(xt0, z, zt, params)
    eqn3 = init_1 - ut0_pred
    return jnp.mean(eqn**2 + eqn2**2 + eqn3**2)

  return jax.value_and_grad(loss)(params, t, z, zt)

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
      zt0 = z_batch[:,1:2].numpy()

      loss, grads = train_step(params, des_batch.numpy(), z0, zt0)
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
  
  global t_0, init_0, init_1
  t_0 = float(t[0])
  init_0 = float(inits[0])
  init_1 = float(inits[1])

  (de_points, zsensors) = defineCollocationPoints(t, N_sensors, sensor_range)

  params = deeponet.init(jax.random.PRNGKey(0), jnp.ones((10,)), jnp.ones((10,)), jnp.ones((10,)))

  params, loss = train_network(params, de_points, zsensors, epochs)

  t = np.linspace(t[0], t[1], eval_points) 
  # Sample a new IC
  init_data1 = inits[0]+0*np.random.uniform(-1.0, 1.0, size=(1,))
  init_data2 = inits[1]+0*np.random.uniform(-1.0, 1.0, size=(1,))
  z = np.repeat(np.expand_dims(init_data1, axis=0), len(t), axis=0)
  zt = np.repeat(np.expand_dims(init_data2, axis=0), len(t), axis=0)

  solPred = deeponet.apply(params, t, z, zt) 

  return loss, t, solPred, params, deeponet, de_points, zsensors
