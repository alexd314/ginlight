from argparse import ArgumentError
from collections import namedtuple
import typing as tp
from   infrastructure.utils.utils import isiterable
import infrastructure.utils.objpath as objpath

InStreamMeta = str
OutStreamMeta = namedtuple('OutStreamMeta',('path','key'))

class Component:

    def _construct_input_stream_list(self, streams : tp.Union[tp.Sequence[str], str, None]) -> tp.Union[None,tp.List[InStreamMeta]]:
        if streams is None:
            in_streams = None
        elif type(streams) == str:
            in_streams = [streams]
        else:
            if isiterable(streams) and all((type(s) is str for s in streams)):
                in_streams = list(streams)
            else:
                raise Exception("Invalid input stream")
        return in_streams

    def _construct_output_stream_list(self, streams : tp.Union[tp.Sequence[tp.Tuple[str,str]],tp.Tuple[str,str],str,None]) -> tp.Union[None, tp.List[OutStreamMeta]]:
        if streams is None:
            out_streams = None
        elif type(streams) == str:
            out_streams = [OutStreamMeta('value',streams)]
        elif type(streams) is tuple:
            out_streams = [OutStreamMeta(streams[0],streams[1])]
        else:
            if isiterable(streams) and all((type(s) is tuple for s in streams)):
                out_streams = list(map(lambda x: OutStreamMeta(x[0],x[1]),streams))
            elif isiterable(streams) and all((type(s) is str for s in streams)):
                out_streams = list(map(lambda x: OutStreamMeta('value',x),streams))
        return out_streams

    def __init__(self,
        input_streams : tp.Union[tp.Sequence[str], str, None],                                      # list of (paths in registry)
        output_streams : tp.Union[tp.Sequence[tp.Tuple[str,str]], tp.Tuple[str,str], str, None],    # list of (path, key)
        core : tp.Callable = None,
        unpack_inputs : bool = True):

        self._input_streams = self._construct_input_stream_list(input_streams)
        self._output_streams = self._construct_output_stream_list(output_streams)
        self._data_registry = None
        self._core = core
        self._core_has_fwd = hasattr(core,'forward')
        self._core_has_call = hasattr(core,'__call__')
        self._unpack_inputs = unpack_inputs

        if not self._core_has_fwd and not self._core_has_call:
            raise ArgumentError("core","Core does not implement forward and is not callable.")

    def _set_output(self, out):

        if self._output_streams is None:
            return

        for path,outkey in self._output_streams:
            value = objpath.get_value(out, path, root = 'value')
            self._data_registry[outkey] = value

    def _get_input(self):

        inp = []
        if self._input_streams is None:
            return inp

        for path in self._input_streams:
            value = objpath.get_value(self._data_registry, path)
            inp.append(value)

        return inp

    def __call__(self):

        if self._core is None:
            return

        inputs = self._get_input()
        if self._core_has_fwd:
            if self._unpack_inputs:
                out = self._core.forward(*inputs)
            else:
                out = self._core.forward(inputs)
        elif self._core_has_call:
            if self._unpack_inputs:
                out = self._core(*inputs)
            else:
                out = self._core(inputs)
        else:
            raise Exception('Core does not implement "forward" and is not callable')

        self._set_output(out)

    # setter injection
    def set_data_registry(self, data_registry):
        self._data_registry = data_registry

    @property
    def core(self):
        return self._core

class Pipeline:

    def __init__(self, components : tp.Sequence[Component]):
        self._components = list(iter(components))

    # setter injection
    def set_data_registry(self, data_registry):
        for c in self._components:
            c.set_data_registry(data_registry)

    def process(self):
        for c in self._components:
            c()

    def __len__(self) -> int:
        return len(self._components)

    def __getitem__(self,index : int) -> Component:
        return self._components[index]