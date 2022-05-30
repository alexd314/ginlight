import gin
import torch
import torch.utils.data as data
import torch.optim
import pytorch_lightning as pl
import torchmetrics
import torchvision
####
def create_tensor(data, dtype=None):
    return torch.tensor(data,dtype = dtype)

########################################################################
# pytorch and pytorch lightning configurables
gin.external_configurable(pl.Trainer)
gin.external_configurable(torch.optim.SGD)
gin.external_configurable(torch.optim.Adam)
gin.external_configurable(torch.optim.lr_scheduler.ReduceLROnPlateau)
gin.external_configurable(pl.loggers.tensorboard.TensorBoardLogger)
gin.external_configurable(data.DataLoader)
gin.external_configurable(pl.callbacks.model_checkpoint.ModelCheckpoint)
gin.external_configurable(torchvision.transforms.RandomCrop, module='torchvision.transforms')
gin.external_configurable(torchvision.transforms.CenterCrop, module='torchvision.transforms')
gin.external_configurable(torchvision.transforms.Compose, module='torchvision.transforms')
gin.external_configurable(torchvision.transforms.ToTensor, module='torchvision.transforms')
gin.external_configurable(torchmetrics.Accuracy, module='torchmetrics')
gin.external_configurable(create_tensor)
##########################################################################
