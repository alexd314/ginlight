import typing as tp
import torch
import torch.nn as nn
import pytorch_lightning as pl
from collections import OrderedDict
from pytorch_lightning import LightningModule
from infrastructure.pipeline import Pipeline
from infrastructure.base.factory import Factory

class LightMain(LightningModule):

    def __init__(self,
        modules : tp.Sequence[nn.Module],
        # training parameters
        optimizer_factory : Factory = None,
        train_pipeline : tp.Union[Pipeline, tp.List[Pipeline]] = None,
        train_batch_input_keys : tp.Union[str,tp.List[str]] = None,
        train_loss_return_keys : tp.Union[str,tp.List[str]] = 'total_loss',
        # validation parameters
        val_pipeline : Pipeline = None,
        val_batch_input_key : str = None,
        val_return_key : str = None,
        # test parameters
        test_pipeline : Pipeline = None,
        test_batch_input_key : str = None,
        test_return_key : str = None,
        # other parameters
        data_registry : tp.Union[None,tp.Dict[str,tp.Any]] = None,
        global_rng_seed : int = None):

        super().__init__()

        self._modulelist = nn.ModuleList(modules)
        self._optimizer_factory = optimizer_factory

        self._train_pipelines = train_pipeline if isinstance(train_pipeline, list) else [train_pipeline]
        self._train_batch_input_keys = train_batch_input_keys if isinstance(train_batch_input_keys,list) else [train_batch_input_keys]
        self._train_loss_return_keys = train_loss_return_keys if isinstance(train_loss_return_keys,list) else [train_loss_return_keys]

        self._val_pipeline = val_pipeline
        self._val_batch_input_key = val_batch_input_key
        self._val_return_key = val_return_key

        self._test_pipeline = test_pipeline
        self._test_batch_input_key = test_batch_input_key
        self._test_return_key = test_return_key

        self._global_rng_seed = global_rng_seed
        self._data_registry = { } if data_registry is None else data_registry

        for p in filter(lambda x: x is not None, self._train_pipelines + [self._val_pipeline, self._test_pipeline]):
            p.set_data_registry(self._data_registry)

        self._seed()

    def _seed(self):
        pl.utilities.seed.seed_everything(seed=self._global_rng_seed)

    def configure_optimizers(self):
        optim = self._optimizer_factory.create()
        
        self._optimizer_step_frequency = { }
        for optimizer_idx in range(len(optim[1])):
            self._optimizer_step_frequency[optimizer_idx] = optim[1][optimizer_idx]

        return optim[0]

    def _move_pipeline_to_device(self, pipeline : Pipeline):
        for c in pipeline:
            if isinstance(c.core, nn.Module):
                c.core.to(self.device)

    # # in lightning, forward contains inference code
    # def forward(self, x : torch.Tensor):
    #     return self._model.forward(x)

    def on_fit_start(self) -> None:
        # move pipeline to device
        for train_pipeline in self._train_pipelines:
            self._move_pipeline_to_device(train_pipeline)

    def on_validation_start(self) -> None:
        # move pipeline to device
        self._move_pipeline_to_device(self._val_pipeline)

    def on_test_start(self) -> None:
        # move pipeline to device
        self._move_pipeline_to_device(self._test_pipeline)

    def _step(self, batch : tp.Dict[str,tp.Any], batch_idx : int, pipeline : Pipeline, batch_input_key : str, return_key : tp.Union[str, None]):
        self._data_registry[batch_input_key] = batch
        pipeline.process()

        ret_val = self._data_registry[return_key] if return_key is not None else None
        return ret_val

    def training_step(self, batch : tp.Dict[str,tp.Any], batch_idx : int, optimizer_idx : int = 0):
        train_pipeline = self._train_pipelines[optimizer_idx]
        
        loss_return_key = self._train_loss_return_keys[optimizer_idx]

        loss = self._step(batch, batch_idx, train_pipeline, self._train_batch_input_keys[optimizer_idx], loss_return_key)
        
        self.log(loss_return_key, loss.item(), prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch : tp.Dict[str,tp.Any], batch_idx : int):
        ret_val = self._step(batch, batch_idx, self._val_pipeline, self._val_batch_input_key, self._val_return_key)        
        return ret_val

    def test_step(self, batch : tp.Dict[str,tp.Any], batch_idx : int):
        ret_val = self._step(batch, batch_idx, self._test_pipeline, self._test_batch_input_key, self._test_return_key)
        return ret_val

    # Alternating schedule for optimizer steps (i.e.: GANs)
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
       
        optimizer_step_frequency = self._optimizer_step_frequency[optimizer_idx]
        if (epoch + batch_idx + 1) % optimizer_step_frequency == 0:     # add epoch bias to make sure at each epoch a different batch is being presented to the pipeline
            optimizer.step(closure = optimizer_closure)
        else:
            optimizer_closure()
        