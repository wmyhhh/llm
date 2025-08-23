_REGISTRY = {}

def register_quantizer(name):
    def wrapper(cls):
        _REGISTRY[name] = cls
        return cls
    return wrapper

def get_quantizer(name):
    if name not in _REGISTRY:
        raise ValueError(f"Quantizer {name} not found. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]