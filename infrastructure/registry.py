import typing as tp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class DataRegistryCleaner(Callback):

    def __init__(self, data_registry : tp.Dict[str,tp.Any]):
        self._data_registry = data_registry

    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: tp.Any, batch_idx: int, unused: int = 0) -> None:
        self._data_registry.clear()

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: tp.Any, batch: tp.Any, batch_idx: int, unused: int = 0) -> None:
        self._data_registry.clear()

    def on_validation_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: tp.Any, batch_idx: int, dataloader_idx: int) -> None:
        self._data_registry.clear()

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: tp.Optional[tp.Any], batch: tp.Any, batch_idx: int, dataloader_idx: int) -> None:
        self._data_registry.clear()

    def on_test_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: tp.Any, batch_idx: int, dataloader_idx: int) -> None:
        self._data_registry.clear()

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: tp.Optional[tp.Any], batch: tp.Any, batch_idx: int, dataloader_idx: int) -> None:
        self._data_registry.clear()