import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import gin
import typing as tp
from infrastructure.utils.utils import *
import os

class GinConfigLogger(Callback):

    def __init__(self, outfile_path : str):
        self._outfile_path = outfile_path
        self._config_written = False

    def _log(self):

        dirname = os.path.dirname(self._outfile_path)
        os.makedirs(dirname,exist_ok=True)

        data = gin.operative_config_str()
        with open(self._outfile_path,'w') as f:
            f.write(data)

    def setup(self, trainer, pl_module, stage = None):
        if not self._config_written:
            self._log()
            self._config_written = True


class ObjectTreeGetter:

    def __init__(self, object : tp.Any, tag : str):
        self._object = object
        self._tag = tag

    def _log_value(self, obj : tp.Any) -> tp.Dict[str,tp.Any]:
        # check for builtin types
        if isbuiltin(obj):
            return obj
        elif istensor(obj):
            return str(obj)
        elif isnamedtupleinstance(obj):
            log_obj = {}
            d = obj._asdict()
            for name in d.keys():
                log_obj[name] = self._log_value(d[name])
            return log_obj
        elif isclass(obj):
            log_obj = {}
            obj_type = type(obj)
            log_obj['class_type'] = str(obj_type)

            for name in get_class_constructor_params(obj):
                if hasattr(obj,name):
                    log_obj[name] = self._log_value(getattr(obj,name))
                elif hasattr(obj,"_"+name):
                    log_obj[name] = self._log_value(getattr(obj,"_"+name))
            return log_obj

        elif isiterable(obj):
            items = list()
            for o in obj:
                items.append(self._log_value(o))
            return items
        else:
            raise Exception("Log value error: Unknown object type")

    def log(self):
        return {self._tag : self._log_value(self._object)}

class ObjectTreeLogger(Callback):

    def __init__(self, objects : tp.List[tp.Any], object_names : tp.List[str], out_file : str):
        assert len(objects) == len(object_names)
        self._objects = objects
        self._object_names = object_names
        self._out_file = out_file
        self._file_written = False

    def setup(self, trainer, pl_module, stage = None):
        if not self._file_written:
            items = { }
            for obj, name in zip(self._objects,self._object_names):
                if obj is not None:
                    tree_getter = ObjectTreeGetter(obj,name)
                    items.update(tree_getter.log())
                else:
                    items[name] = obj
            write_json(items, self._out_file)
            self._file_written = True
