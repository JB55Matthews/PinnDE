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

def pinnSelector(key):
    return FUNCTION_REGISTRY.get(key, lambda: print("Invalid key"))