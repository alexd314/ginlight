import typing as tp
import pandas as pd
import os
from operator import itemgetter, lt, gt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from utils.metrics import MetricProcessor

class MetricFileLogger(Callback):

    _reduce = {
        'min' : min,
        'max' : max
    }
    _comp = {
        'min' : lt,
        'max' : gt
    }

    def __init__(self, out_log_file : str,
                       data_registry : tp.Dict[str,tp.Any],
                       registry_keys : tp.List[str],
                       append : bool = True,
                       flush_always : bool = True,
                       stages = tp.List[str],
                       experiment_name : tp.Union[str,None] = None):

        self._out_log_file = out_log_file
        self._data_registry = data_registry
        self._registry_keys = registry_keys
        self._append = append
        self._flush_always = flush_always
        self._stages = stages
        self._experiment_name = experiment_name if experiment_name is not None else "N/A"
        self._metrics = [ ]

    def _update_log(self, value : tp.List[tp.Dict[str,tp.Any]]):

        os.makedirs(os.path.dirname(self._out_log_file),exist_ok=True)

        if self._append and os.path.exists(self._out_log_file):
            df = pd.read_csv(self._out_log_file)
            df = pd.concat([df,pd.DataFrame(value)],ignore_index=True)
        else:
            df = pd.DataFrame(value)

        df.to_csv(self._out_log_file, index=False)

    def _on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        if trainer.sanity_checking:
            return

        epoch_number = pl_module.current_epoch

        metric_data = {
            "experiment" : self._experiment_name,
            "epoch" : epoch_number
        }
        for regkey in self._registry_keys:
            if not regkey.startswith("*"):
                value = self._data_registry.get(regkey,None)
                if value is not None:
                    metric_data[regkey] = value
            else:
                key_pattern = regkey[1:]   # skip *
                # TODO: do regex here
                for preg_key in filter(lambda key: key.endswith(key_pattern),self._data_registry.keys()):
                    value = self._data_registry.get(preg_key, None)
                    if value is not None:
                        if isinstance(value,list):
                            for idx, v in enumerate(value):
                                metric_data[preg_key+"_"+str(idx)] = v
                        else:
                            metric_data[preg_key] = value

        self._metrics.append(metric_data)

        if self._flush_always:
            self._update_log(self._metrics)
            self._metrics.clear()

    def _on_stage_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        if trainer.sanity_checking:
            return

        if len(self._metrics) > 0:
            self._update_log(self._metrics)
            self._metrics.clear()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if 'val' in self._stages:
            self._on_epoch_end(trainer, pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if 'val' in self._stages:
            self._on_stage_end(trainer, pl_module)

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if 'test' in self._stages:
            self._on_epoch_end(trainer, pl_module)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if 'test' in self._stages:
            return self._on_stage_end(trainer,pl_module)