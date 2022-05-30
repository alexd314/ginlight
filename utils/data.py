from pytorch_lightning import LightningDataModule
import torch.utils.data as data

class DataModule(LightningDataModule):

    def __init__(self, dataset : data.Dataset,
                       train_dataloader_cls : data.DataLoader.__class__,
                       val_dataloader_cls : data.DataLoader.__class__,
                       test_dataloader_cls : data.DataLoader.__class__
                       ):
        super().__init__()
        self._dataset = dataset

        self._train_dataloader_cls = train_dataloader_cls
        self._val_dataloader_cls = val_dataloader_cls
        self._test_dataloader_cls = test_dataloader_cls

    def setup(self, stage : str):
        if hasattr(self._dataset,'setup'):
            self._dataset.setup(stage)

    def train_dataloader(self):
        if self._dataset.train_dataset() is not None and len(self._dataset.train_dataset()) > 0:
            return self._train_dataloader_cls(dataset = self._dataset.train_dataset())
        else:
            return None

    def test_dataloader(self):
        if self._dataset.test_dataset() is not None  and len(self._dataset.test_dataset()) > 0:
            return self._test_dataloader_cls(dataset = self._dataset.test_dataset())
        else:
            return None

    def val_dataloader(self):
        if self._dataset.val_dataset() is not None and len(self._dataset.val_dataset()) > 0:
            return self._val_dataloader_cls(self._dataset.val_dataset())
        else:
            return None