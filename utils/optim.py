import torch.nn as nn
import typing as tp
import torch.optim as optim
from infrastructure.base.factory import Factory

class StandardOptimizerFactory(Factory):

    def __init__(self,
                optimizer_cls,
                module : nn.Module,
                lr_scheduler_cls = tp.Union[None, optim.Optimizer.__class__],
                lr_scheduler_interval: tp.Union[int,None] = None,
                lr_scheduler_frequency: tp.Union[int, None] = None,
                lr_scheduler_monitor_metric : tp.Union[str, None] = None,
                singleton : bool = True):

        self._optimizer_cls = optimizer_cls
        self._module = module
        self._lr_scheduler_cls = lr_scheduler_cls
        self._lr_scheduler_interval = lr_scheduler_interval
        self._lr_scheduler_frequency = lr_scheduler_frequency
        self._lr_scheduler_monitor_metric = lr_scheduler_monitor_metric
        self._singleton = singleton
        self._singleton_value = None

    def create(self) -> tp.Dict[str,tp.Any]:

        if self._singleton and self._singleton_value is not None:
            return self._singleton_value

        optimizer = self._optimizer_cls(self._module.parameters())

        ret = {
            'optimizer' : optimizer
        }

        if self._lr_scheduler_cls is not None and self._lr_scheduler_monitor_metric is not None:
            lr_scheduler_cfg = [
                (k,v) for k,v in filter(lambda x: x[1] is not None, [
                    ('scheduler', self._lr_scheduler_cls(optimizer)),
                    ('interval', self._lr_scheduler_interval),
                    ('frequency', self._lr_scheduler_frequency),
                    ('monitor', self._lr_scheduler_monitor_metric)])
            ]
            ret.update(lr_scheduler_cfg)

        if self._singleton:
            self._singleton_value = ret

        return ret, (1,)       # update every 1 step


class MultipleOptimizerFactory(Factory):

    def __init__(self, optim_factories : tp.List[StandardOptimizerFactory], update_every_steps : tp.List[int]):
        assert len(optim_factories) == len(update_every_steps)
        self._optim_factories = optim_factories
        self._update_every_steps = update_every_steps

    def create(self) -> tp.Tuple[tp.Dict[str,tp.Any]]:

        ret = list(map(lambda optim: optim.create()[0], self._optim_factories)), tuple(self._update_every_steps)
        return ret