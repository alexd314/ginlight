import typing as tp
from enum import Enum
from collections import deque
import infrastructure.utils.iter as itertoolsext

class PARSE_ACTION (Enum):
    PARSE_ATTRIBUTE = 0,
    PARSE_INDEXER = 1

def _get_indexer(obj : tp.Any, attribute : str):
    if(hasattr(obj,'__getitem__')):
        try:
            return obj[attribute]
        except:
            try:
                index = int(attribute)
                return obj[index]
            except Exception as ex:
                raise ex

def _get_attribute(obj : tp.Any, attribute : str):
    if(hasattr(obj,attribute)):
        return getattr(obj,attribute)
    else:
        return _get_indexer(obj,attribute)

def _get_value_in_hierarchy(obj : tp.Any, path : str):

    actions = {
        PARSE_ACTION.PARSE_ATTRIBUTE : _get_attribute,
        PARSE_ACTION.PARSE_INDEXER : _get_indexer
    }

    lastchar = deque(maxlen=1)
    current = iter(path)
    finished = False
    next_action = PARSE_ACTION.PARSE_ATTRIBUTE
    next_obj = obj
    while not finished:

        c = "".join(itertoolsext.takewhile_withlast(lambda x: x not in ('.','[',']'), current, lastchar))
        c = c.strip()
        if len(c)>0:
            next_obj = actions[next_action](next_obj,c)
        if len(lastchar) == 0:
            finished = True
        else:
            char = lastchar.pop()
            if(char == '.'):
                next_action = PARSE_ACTION.PARSE_ATTRIBUTE
            else:
                next_action = PARSE_ACTION.PARSE_INDEXER

    return next_obj

def get_value(obj : tp.Any, path : str, root : tp.Union[None,str] = None):

    if root is not None:
        assert path.startswith(root)
        path = path[len(root):]

    if len(path) == 0:
        return obj
    else:
        return _get_value_in_hierarchy(obj, path)