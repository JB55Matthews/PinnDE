import tensorflow as tf

FUNCTION_REGISTRY = {}

def register(key):
    def decorator(func):
        FUNCTION_REGISTRY[key] = func
        return func
    return decorator

@register("hard")
def isHardConstrained():
    return True

@register("soft")
def isHardConstrained():
    return False

@register("adam")
def adam(lr):
    return tf.keras.optimizers.Adam(lr)

@register("adagrad")
def adagrad(lr):
    return tf.kerars.optimizers.Adagrad(lr)

@register("sgd")
def sgd(lr):
    return tf.keras.optimizers.SGD(lr)

@register("rmsprop")
def rmsprop(lr):
    return tf.keras.optimizers.RMSprop(lr)

def pinnSelector(key):
    return FUNCTION_REGISTRY.get(key, lambda: print("Invalid key"))