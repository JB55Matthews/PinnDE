import tensorflow as tf


# Define the network
def build_model(nr_layers, nr_units, summary=True):
  """
  Builds standard PINN in tensorflow

  Args:
    nr_layers (int): Number of internal layer of network
    nr_units (int): Number of nodes per internal layer of network
    summary (bool): decides whether summary of network is printed

  Returns:
    model (PINN): Constructed model
  """

  inp = b = tf.keras.layers.Input(shape=(1,))

  for i in range(nr_layers):
    b = tf.keras.layers.Dense(nr_units, activation='tanh')(b)
  out = tf.keras.layers.Dense(1, activation='linear')(b)

  model = tf.keras.models.Model(inp, out)

  if summary:
    model.summary()

  return model

def build_model_hardConstraint_order1_IVP(inits, nr_layers, nr_units, summary=True):
  """
  Builds PINN with hard constrainted inital values for first order ODE in tensorflow

  Args:
    inits (list): List containing inital values
    nr_layers (int): Number of internal layer of network
    nr_units (int): Number of nodes per internal layer of network
    summary (bool): decides whether summary of network is printed

  Returns:
    model (PINN): Constructed model
  """

  inp = b = tf.keras.layers.Input(shape=(1,))

  for i in range(nr_layers):
    b = tf.keras.layers.Dense(nr_units, activation='tanh')(b)
  out = tf.keras.layers.Dense(1, activation='linear')(b)

  out = tf.keras.layers.Lambda(lambda x: inits[2] + ((x[0]-inits[0])/(inits[1]-inits[0]))*x[1])([inp, out])

  model = tf.keras.models.Model(inp, out)

  if summary:
    model.summary()

  return model

def build_model_hardConstraint_order2_IVP(inits, nr_layers, nr_units, summary=True):
  """
  Builds PINN with hard constrainted inital values for second order ODE in tensorflow

  Args:
    inits (list): List containing inital values
    nr_layers (int): Number of internal layer of network
    nr_units (int): Number of nodes per internal layer of network
    summary (bool): decides whether summary of network is printed

  Returns:
    model (PINN): Constructed model
  """

  inp = b = tf.keras.layers.Input(shape=(1,))

  for i in range(nr_layers):
    b = tf.keras.layers.Dense(nr_units, activation='tanh')(b)
  out = tf.keras.layers.Dense(1, activation='linear')(b)

  out = tf.keras.layers.Lambda(lambda x: inits[2] + (x[0]-inits[0])*inits[3] + ((x[0]-inits[0])/(inits[1]-inits[0]))**2*x[1])([inp, out])

  model = tf.keras.models.Model(inp, out)

  if summary:
    model.summary()

  return model


def build_model_hardConstraint_order3_IVP(inits, nr_layers, nr_units, summary=True):
  """
  Builds PINN with hard constrainted inital values for third order ODE in tensorflow

  Args:
    inits (list): List containing inital values
    nr_layers (int): Number of internal layer of network
    nr_units (int): Number of nodes per internal layer of network
    summary (bool): decides whether summary of network is printed

  Returns:
    model (PINN): Constructed model
  """

  inp = b = tf.keras.layers.Input(shape=(1,))

  for i in range(nr_layers):
    b = tf.keras.layers.Dense(nr_units, activation='tanh')(b)
  out = tf.keras.layers.Dense(1, activation='linear')(b)

  out = tf.keras.layers.Lambda(lambda x: inits[2] + (x[0]-inits[0])*inits[3] + 
                               ((x[0]-inits[0])**2)*inits[4] + (((x[0]-inits[0])/(inits[1]-inits[0]))**3)*x[1])([inp, out])


  model = tf.keras.models.Model(inp, out)

  if summary:
    model.summary()

  return model

def build_model_hardConstraint_order4_IVP(inits, nr_layers, nr_units, summary=True):
  """
  Builds PINN with hard constrainted inital values for fourth order ODE in tensorflow

  Args:
    inits (list): List containing inital values
    nr_layers (int): Number of internal layer of network
    nr_units (int): Number of nodes per internal layer of network
    summary (bool): decides whether summary of network is printed

  Returns:
    model (PINN): Constructed model
  """

  inp = b = tf.keras.layers.Input(shape=(1,))

  for i in range(nr_layers):
    b = tf.keras.layers.Dense(nr_units, activation='tanh')(b)
  out = tf.keras.layers.Dense(1, activation='linear')(b)

  out = tf.keras.layers.Lambda(lambda x: inits[2] + (x[0]-inits[0])*inits[3] +
                              ((x[0]-inits[0])**2)*inits[4] + ((x[0]-inits[0])**3)*inits[5] +
                              (((x[0]-inits[0])/(inits[1]-inits[0]))**4)*x[1])([inp, out])


  model = tf.keras.models.Model(inp, out)

  if summary:
    model.summary()

  return model

def build_model_hardConstraint_order5_IVP(inits, nr_layers, nr_units, summary=True):
  """
  Builds PINN with hard constrainted inital values for fifth order ODE in tensorflow

  Args:
    inits (list): List containing inital values
    nr_layers (int): Number of internal layer of network
    nr_units (int): Number of nodes per internal layer of network
    summary (bool): decides whether summary of network is printed

  Returns:
    model (PINN): Constructed model
  """

  inp = b = tf.keras.layers.Input(shape=(1,))

  for i in range(nr_layers):
    b = tf.keras.layers.Dense(nr_units, activation='tanh')(b)
  out = tf.keras.layers.Dense(1, activation='linear')(b)

  out = tf.keras.layers.Lambda(lambda x: inits[2] + (x[0]-inits[0])*inits[3] +
                              ((x[0]-inits[0])**2)*inits[4] + ((x[0]-inits[0])**3)*inits[5] + ((x[0]-inits[0])**4)*inits[6] +
                              (((x[0]-inits[0])/(inits[1]-inits[0]))**5)*x[1])([inp, out])


  model = tf.keras.models.Model(inp, out)

  if summary:
    model.summary()

  return model



def build_model_hardConstraint_order12_BVP(inits, nr_layers, nr_units, summary=True):
  """
  Builds PINN with hard constrainted boundary values for a first or second order ODE in tensorflow

  Args:
    inits (list): List containing inital boundary values
    nr_layers (int): Number of internal layer of network
    nr_units (int): Number of nodes per internal layer of network
    summary (bool): decides whether summary of network is printed

  Returns:
    model (PINN): Constructed model
  """


  inp = b = tf.keras.layers.Input(shape=(1,))

  for i in range(nr_layers):
    b = tf.keras.layers.Dense(nr_units, activation='tanh')(b)
  out = tf.keras.layers.Dense(1, activation='linear')(b)

  out = tf.keras.layers.Lambda(lambda x: inits[2]*(1-((x[0]-inits[0])/(inits[1]-inits[0]))) 
                               + inits[3]*((x[0]-inits[0])/(inits[1]-inits[0]))
                          + ((x[0]-inits[0])/(inits[1]-inits[0]))*(1-((x[0]-inits[0])/(inits[1]-inits[0])))*x[1])([inp, out])

  #SHOULD LOOK LIKE:
  #when x = leftboundry
  #A(1) + B(0) + (0)(somehting)(x)
  #when x = rightboundry
  #A(0) + B(1) + (something)(0)(x)

  #A(1-((x-left)/(right-left)))) + B*(x-left)/(right-left)) + (x-left)/(right-left))(1-((x-left)/(right-left))))*x[1]



  model = tf.keras.models.Model(inp, out)

  if summary:
    model.summary()

  return model

def build_model_hardConstraint_order3_BVP(inits, nr_layers, nr_units, summary=True):
  """
  Builds PINN with hard constrainted boundary values for third order ODE in tensorflow

  Args:
    inits (list): List containing inital boundary values
    nr_layers (int): Number of internal layer of network
    nr_units (int): Number of nodes per internal layer of network
    summary (bool): decides whether summary of network is printed

  Returns:
    model (PINN): Constructed model
  """

  inp = b = tf.keras.layers.Input(shape=(1,))

  for i in range(nr_layers):
    b = tf.keras.layers.Dense(nr_units, activation='tanh')(b)
  out = tf.keras.layers.Dense(1, activation='linear')(b)

  out = tf.keras.layers.Lambda(lambda x: inits[2]*(1-((x[0]-inits[0])/(inits[1]-inits[0]))) 
                               + inits[3]*((x[0]-inits[0])/(inits[1]-inits[0]))
                    + (inits[4]+inits[2]-inits[3])*((1-((x[0]-inits[0])/(inits[1]-inits[0])))**2)*((x[0]-inits[0])/(inits[1]-inits[0]))
                    - (inits[5]+inits[2]-inits[3])*(((x[0]-inits[0])/(inits[1]-inits[0]))**2)*(1-((x[0]-inits[0])/(inits[1]-inits[0])))
                          +(((x[0]-inits[0])/(inits[1]-inits[0]))**2)*((1-((x[0]-inits[0])/(inits[1]-inits[0])))**2)*x[1])([inp, out])

  model = tf.keras.models.Model(inp, out)

  #A(1-x) + Bx + | C(1-x)^2x + D(1-x)x^2 + (1-x)^2 (x)^2 out | 

  #DER: -A + B  + C(1-x)^2 -2Cx(1-x) - Dx^2 + 2D(1-x)x - x^2 2(1-x) out + out(1-x)^2 2x
  #offset of B - A to every calcualtoion
  #make  C and D into C or D + A - B
  #Also, d/dx(1-x) =-1, so we need -(D+A-B)

 #A(1-x) + Bx +  (C+A-B)x(1-x)^2 - (D+A-B)(1-x)x^2 + (1-x)^2 (x)^2 out
 #when x = leftboundry (x=0)
 # A + 0 + 0 + 0 + 0 = A
 # x = right (x=1)
 # 0 + B + 0 + 0 + 0 = B
 #DER: -A + B + (C+A-B)(1-x)^2 - 2x(C+A-B)(1-x) - 2x(D+A-B)(1-x) + (D+A-B)x^2 + 2x(1-x)^2 out - 2(1-x)x^2 out
 #when x = left (x=0)
 # -A + B + (C+A-B) - 0 - 0 + 0 + 0 - 0 = C
 #when x = right (x=1)
 # -A + B + 0 - 0 - 0 + D+A-B + 0 - 0 = D
  

  if summary:
    model.summary()

  return model


def build_model_system2_IVP(nr_layers, nr_units, summary=True):
  """
  Builds PINN for solving two equations in tensorflow

  Args:
    nr_layers (int): Number of internal layer of network
    nr_units (int): Number of nodes per internal layer of network
    summary (bool): decides whether summary of network is printed

  Returns:
    model (PINN): Constructed model
  """

  inp = b = tf.keras.layers.Input(shape=(1,))

  for i in range(nr_layers):
    b = tf.keras.layers.Dense(nr_units, activation='tanh')(b)
  outu = tf.keras.layers.Dense(1, activation='linear')(b)
  outx = tf.keras.layers.Dense(1, activation='linear')(b)

  model = tf.keras.models.Model(inp, [outu, outx])

  if summary:
    model.summary()

  return model

def build_model_system3_IVP(nr_layers, nr_units, summary=True):
  """
  Builds PINN for solving three equations in tensorflow

  Args:
    nr_layers (int): Number of internal layer of network
    nr_units (int): Number of nodes per internal layer of network
    summary (bool): decides whether summary of network is printed

  Returns:
    model (PINN): Constructed model
  """

  inp = b = tf.keras.layers.Input(shape=(1,))

  for i in range(nr_layers):
    b = tf.keras.layers.Dense(nr_units, activation='tanh')(b)
  outu = tf.keras.layers.Dense(1, activation='linear')(b)
  outx = tf.keras.layers.Dense(1, activation='linear')(b)
  outy = tf.keras.layers.Dense(1, activation='linear')(b)

  model = tf.keras.models.Model(inp, [outu, outx, outy])

  if summary:
    model.summary()

  return model