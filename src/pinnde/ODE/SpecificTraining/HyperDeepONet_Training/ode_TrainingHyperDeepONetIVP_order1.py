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

# Convert a pytree to a 1d numpy array
def flatten_pytree(params):
  values, tree_info = jax.tree_util.tree_flatten(params)

  no_leaves = len(values)
  shapes = []
  flattened = []

  for i in range(no_leaves):
    flattened.append(values[i].flatten())
    shapes.append(values[i].shape)

  return tree_info, shapes, jnp.concatenate(flattened)

# Get a pytree from a 1d numpy array
def reconstruct_pytree(tree_info, shapes, flattened):

  values = []
  k = 0
  for i in range(len(shapes)):
    values.append(np.reshape(flattened[k:k+np.prod(shapes[i])], shapes[i]))
    k += np.prod(shapes[i])

  py_tree = jax.tree_util.tree_unflatten(tree_info, values)
  return py_tree

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
  units_out: int

  @nn.compact
  def __call__(self, inp):
    b = inp
    for i in range(self.layers-1):
      b = nn.Dense(self.units)(b)
      b = nn.tanh(b)
    out = nn.Dense(self.units_out)(b)
    return out
  
class HardConstraint(nn.Module):
  t0: float
  tfinal: float
  u0: float

  @nn.compact
  def __call__(self, inputs):
    t, nn = inputs
    return self.u0 + (t-self.t0)/(self.tfinal-self.t0)*nn
  
class TrunkNet(nn.Module):

  t0: float
  tfinal: float
  layers: int
  units: int

  @nn.compact
  def __call__(self, t):

    b = Normalize(self.t0, self.tfinal)(t)
    net = MLP(self.layers, self.units, self.units)(b)
    q = nn.Dense(1)(net)

    return q
  

class HyperBranchNet(nn.Module):

  layers: int
  units: int
  lora_rank: int
  out_units: int

  @nn.compact
  def __call__(self, u):

    net = MLP(self.layers, self.units, self.units)(u)

    if self.lora_rank == -1: # Use full hypernet
      out = nn.Dense(self.out_units)(net)
    else: # Use LoRA approximation
      W1 = self.param('W1', nn.initializers.glorot_normal(), (net.shape[1], self.lora_rank))
      W2 = self.param('W2', nn.initializers.glorot_normal(), (self.lora_rank, no_trunk_params))
      W = jnp.matmul(W1, W2)
      out = jnp.matmul(net, W)

    return out
  
class DeepONet(nn.Module):

  t0: float
  tfinal: float
  layers: int
  units: int
  lora_rank: int

  @nn.compact
  def __call__(self, t, u):

    # Add the batch dimension
    if u.ndim == 1:
      u = jnp.reshape(u, (1,-1))
    if t.ndim == 1:
      t = jnp.reshape(t, (-1,1))

    # These are the ICs for the hard constraint
    q0 = u[:,:1]

    z = jnp.concatenate([u], axis=1)

    # Hypernetwork to predict the weights of the trunk net
    params = HyperBranchNet(self.layers, self.units, self.lora_rank, no_trunk_params)(z)
    batch_size = params.shape[0]
    q = []
    t = jnp.reshape(t, (-1,1))
    for i in range(batch_size):
      cparams = reconstruct_pytree(tree_info, shapes, params[i,])
      qc= trunk_net.apply(cparams, t[i:i+1,])
      q.append(qc[:,0])
    q = jnp.array(q)
    q = HardConstraint(self.t0, self.tfinal, q0)([t, q])
    q = jnp.reshape(q, (-1,))
    return q
  
# Define the collocation points over the domain [t_0, t_1]
def defineCollocationPoints(t_bdry, N, sensor_range):

    # Sample points where to evaluate the PDE
  ode_points = t_bdry[0] + (t_bdry[1] - t_bdry[0])*lhs(1, N)
  zsensors = np.random.uniform(float(sensor_range[0]), float(sensor_range[1]), size=(N, 1))

  return (ode_points, zsensors)

# DeepONet train/loss function
@jax.jit
def train_step(params, pdes, z):
  t = pdes[:,0]
  def q_model(t, z, params):
    return deeponet.apply(params, t, z)[0]
  def q_t(t, z, params):
    return jax.vmap(jax.grad(q_model, 0), [0, 0, None])(t, z, params)
  def loss(params, t, z):
    u = deeponet.apply(params, t, z)
    ut = q_t(t, z, params)
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

  nr_batches = 10
  batch_size = N//nr_batches

  lr = 1e-3
  lr = optax.exponential_decay(lr, epochs, 0.9)
  #lr = optax.exponential_decay(lr, 500, 0.9)
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
      print(f'Loss in epoch {i}: {epoch_loss[i]}, Lr: {lr(i):1.4f}')

  return params, epoch_loss

def startTraining(eval_points, inits, order, t, N_sensors, sensor_range, epochs, net_layers, net_units, eqn):

  global equation, deeponet
  equation = eqn
  
  (de_points, zsensors) = defineCollocationPoints(t, N_sensors, sensor_range)
  
  global trunk_net, tree_info, shapes, no_trunk_params
  trunk_net = TrunkNet(t[0], t[1], layers=net_layers, units=net_units+20)

  # Initialize the model parameters
  params = trunk_net.init(jax.random.PRNGKey(0), jnp.ones((10,1))) ####
  tree_info, shapes, params_flattened = flatten_pytree(params)

  no_trunk_params = len(params_flattened)

    # Create the model (LoRA-HyperNet)
  deeponet = DeepONet(t[0], t[1], layers=net_layers, units=net_units, lora_rank=-1)


  # Initialize the model parameters
  params = deeponet.init(jax.random.PRNGKey(0), jnp.ones((1,)), jnp.ones((1,))) ###

  params, loss = train_network(params, de_points, zsensors, epochs)

  
  t = np.linspace(t[0], t[1], eval_points)
  # Sample a new IC
  init_data1 = inits[0]+0*np.random.uniform(-1.0, 1.0, size=(1,))
  z = np.repeat(np.expand_dims(init_data1, axis=0), len(t), axis=0)

  # Model prediction
  solPred = deeponet.apply(params, t, z)


  return loss, t, solPred, params, deeponet, de_points, zsensors
