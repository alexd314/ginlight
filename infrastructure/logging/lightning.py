import typing as tp
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torch.utils.tensorboard.writer import SummaryWriter
from infrastructure.utils.objpath import get_value as get_obj_path_value

class ScalarLogger(Callback):

    def __init__(self, data_registry : tp.Dict[str,tp.Any],
                    metric : tp.Union[str,tp.Sequence[str]],
                    registry_key : tp.Union[str, tp.Sequence[str]]
                    ):

        self._metrics = metric if type(metric) != str else [metric]
        self._registry_keys = registry_key if type(registry_key) != str else [registry_key]
        self._data_registry = data_registry
        assert len(self._metrics) == len(self._registry_keys)

    def setup(self, trainer, pl_module, stage = None):
        self._logger = pl_module

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: tp.Any, batch: tp.Any, batch_idx: int, unused: int = 0) -> None:
        self._log()

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: tp.Any, batch: tp.Any, batch_idx: int, dataloader_idx: int) -> None:
        self._log()

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: tp.Any, batch: tp.Any, batch_idx: int, dataloader_idx: int) -> None:
        self._log()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._log()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._log()

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._log()

    def _log(self):

        for metric, reg_key in zip(self._metrics, self._registry_keys):
            if not reg_key.startswith("*"):
                value = self._data_registry.get(reg_key, None)
                if value is not None:
                    self._logger.log(metric, value)
            else:
                key_pattern = reg_key[1:]   # skip *
                # TODO: do regex here
                for preg_key in filter(lambda key: key.endswith(key_pattern),self._data_registry.keys()):
                    value = self._data_registry.get(preg_key, None)
                    if value is not None:
                        if isinstance(value,list):
                            for idx, v in enumerate(value):
                                self._logger.log(metric+preg_key+"_"+str(idx),v)
                        else:
                            self._logger.log(metric+preg_key,value)


class TensorboardImageLogger(Callback):

    def __init__(self, data_registry : tp.Dict[str,tp.Any],
                        registry_path: str,
                        image_tag : str,
                        image_count : int,
                        stages : tp.Union[str,tp.Sequence[str]],
                        step_interval : int = 10,
                        random : bool = True
                        ):

        self._registry_path = registry_path
        self._data_registry = data_registry
        self._image_tag = image_tag
        self._image_count = image_count
        self._stages = stages
        self._random = random

        self._step = 0
        self._step_interval = step_interval

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: tp.Optional[str] = None) -> None:
        self._logger = pl_module.logger     # get tensorboard logger, assuming lightning module was instantiated with tensorboard logger

    def _log(self, stage : str = ''):

        if self._step % self._step_interval == 0:
            images = get_obj_path_value(self._data_registry, self._registry_path)
            B = images.shape[0]

            if self._random:
                indices = torch.randint(low=0,high=B,size=(self._image_count,))
            else:
                indices = torch.arange(0,self._image_count)

            images = images[indices,...]

            experiment : SummaryWriter = self._logger.experiment
            experiment.add_images(stage+self._image_tag,images,global_step=self._step)

        self._step +=1

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: tp.Any, batch: tp.Any, batch_idx: int, unused: int = 0) -> None:
        if 'train' in self._stages:
            self._log('train-')

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: tp.Optional[tp.Any], batch: tp.Any, batch_idx: int, dataloader_idx: int) -> None:
        if 'val' in self._stages:
            self._log('val-')

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: tp.Optional[tp.Any], batch: tp.Any, batch_idx: int, dataloader_idx: int) -> None:
        if 'test' in self._stages:
            self._log('test-')



class TensorboardImagePairLogger(Callback):

    def __init__(self, data_registry : tp.Dict[str,tp.Any],
                       registry_path1: str,
                       registry_path2: str,
                       image_tag1 : str,
                       image_tag2 : str,
                       image_pair_count : int,
                       stages : tp.Union[str,tp.Sequence[str]],
                       random : bool = True,
                       step_interval : int = 10,
                       ):

        self._data_registry = data_registry
        self._registry_path1 = registry_path1
        self._registry_path2 = registry_path2

        self._image_tag1 = image_tag1
        self._image_tag2 = image_tag2
        self._image_pair_count = image_pair_count
        self._stages = stages
        self._random = random

        self._step = 0
        self._step_interval = step_interval

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: tp.Optional[str] = None) -> None:
        self._logger = pl_module.logger     # get tensorboard logger, assuming lightning module was instantiated with tensorboard logger

    def _log(self, stage : str = ''):

        if self._step % self._step_interval == 0:
            images1 = get_obj_path_value(self._data_registry, self._registry_path1)
            images2 = get_obj_path_value(self._data_registry, self._registry_path2)

            B1 = images1.shape[0]
            B2 = images2.shape[0]
            assert B1 == B2

            B = B1

            if self._random:
                indices = torch.randint(low=0,high=B,size=(self._image_pair_count,))
            else:
                indices = torch.arange(0,self._image_pair_count)

            images1 = images1[indices,...]
            images2 = images2[indices,...]

            experiment : SummaryWriter = self._logger.experiment
            experiment.add_images(stage+self._image_tag1,images1,global_step=self._step)
            experiment.add_images(stage+self._image_tag2,images2,global_step=self._step)

        self._step +=1

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: tp.Any, batch: tp.Any, batch_idx: int, unused: int = 0) -> None:
        if 'train' in self._stages:
            self._log('train-')

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: tp.Optional[tp.Any], batch: tp.Any, batch_idx: int, dataloader_idx: int) -> None:
        if 'val' in self._stages:
            self._log('val-')

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: tp.Optional[tp.Any], batch: tp.Any, batch_idx: int, dataloader_idx: int) -> None:
        if 'test' in self._stages:
            self._log('test-')