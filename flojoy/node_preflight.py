from typing import Callable

class PreflightStore:
    preflight_functions = {}

    @classmethod
    def get_function(cls, node_name: str):
        return cls.preflight_functions.get(node_name)

def node_preflight(for_node: Callable):
    def decorator(func: Callable):
        PreflightStore.preflight_functions[for_node.__name__] = func
        return func

    return decorator
