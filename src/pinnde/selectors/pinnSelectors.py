
FUNCTION_REGISTRY = {}

def register(key):
    def decorator(func):
        FUNCTION_REGISTRY[key] = func
        return func
    return decorator

@register(1)
def func_1():
    print("Function 1 executed")

@register(2)
def func_2():
    print("Function 2 executed")

def pinnSelector(i):
    return FUNCTION_REGISTRY.get(i, lambda: print("Invalid key"))