import typing as tp
import torch
from torchmetrics import Metric
from pytorch_lightning.callbacks import Callback

class MetricProcessor(Callback):

    def __init__(self, metric_cls : Metric.__class__, stages : tp.List[str], data_registry : tp.Dict[str,tp.Any] = None, registry_key : str = None):
        """
        stages: any combination of ['train','val','test']
        """
        self._metric_cls = metric_cls
        self._stages = stages
        self._metric = None
        self._data_registry = data_registry
        self._registry_key = registry_key
        self._cached_metric = None

    def _on_epoch_start(self, stage : str, device : torch.device):
        if stage in self._stages:
            self._metric = self._metric_cls()
            self._metric.to(device)
            self._cached_metric = None

    def _on_epoch_end(self, stage : str):
        if stage in self._stages:
            if self._data_registry is not None and self._registry_key is not None:
                self._cached_metric = self._metric.compute().item()
                self._data_registry[self._registry_key] = self._cached_metric

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._on_epoch_start('train', pl_module.device)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._on_epoch_start('val', pl_module.device)

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self._on_epoch_start('test',pl_module.device)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._on_epoch_end('train')

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._on_epoch_end('val')

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        self._on_epoch_end('test')

    def __call__(self, *args):
        self._metric.update(*args)

    def get_value(self):
        if self._metric is not None:
            if self._cached_metric is not None:
                return self._cached_metric
            else:
                return self._metric.compute()
        else:
            return None


