import typing as tp
import inspect
from torch import Tensor
import os
import json

def isiterable(obj : tp.Any) -> bool:
    try:
        iterator = iter(obj)
    except TypeError as te:
        return False
    return True


def isnamedtupleinstance(x):
    """
    Check for a named tuple
    https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple
    """
    return (
            isinstance(x, tuple) and
            hasattr(x, '_asdict') and
            hasattr(x, '_fields')
    )

def isbuiltin(obj : tp.Any) -> bool:
    obj_type = type(obj)
    return obj_type in [str,int,float,bool,complex,type(None)]

def isclass(obj : tp.Any) -> bool:
    obj_type = type(obj)
    return inspect.isclass(obj_type) and not (obj_type in (list,tuple,str))

def get_class_constructor_params(obj : tp.Any):
    obj_type = type(obj)
    return inspect.signature(obj_type.__init__).parameters

def istensor(obj : tp.Any) -> bool:
    return type(obj) == Tensor


def write_json(obj : tp.Any, file_path : str):
    dirname = os.path.dirname(file_path)
    os.makedirs(dirname,exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=4)
