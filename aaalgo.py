import mercury as mr
import jsonpickle

def inspect (obj):
    if hasattr(obj, 'json') and callable(getattr(obj, 'json')):
        return mr.JSON(obj.json())
    return mr.JSON(jsonpickle.encode(obj))

