import gin
from examples.classification.data.CUBDataset import CUBDataset
from torchvision.models.alexnet import alexnet
from torchvision.models.resnet import resnet18
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.transforms.transforms import RandomApply, ColorJitter, Grayscale, RandomGrayscale, RandomRotation, Compose
from torchvision.transforms.functional import InterpolationMode
from torch.nn import CrossEntropyLoss, Softmax
from torch.utils.data.sampler import RandomSampler

###### datasets
gin.external_configurable(CUBDataset)
###### models
gin.external_configurable(alexnet)
gin.external_configurable(resnet18)
gin.external_configurable(mobilenet_v2)
######
###### torch
gin.external_configurable(CrossEntropyLoss)
gin.external_configurable(Softmax)
gin.external_configurable(RandomApply)
gin.external_configurable(ColorJitter)
gin.external_configurable(Grayscale)
gin.external_configurable(RandomGrayscale)
gin.external_configurable(RandomRotation)
gin.external_configurable(Compose)
gin.external_configurable(RandomSampler)
######

## resolvers
def as_interpolation_mode(mode : str) -> InterpolationMode:
    return InterpolationMode(mode)

gin.external_configurable(as_interpolation_mode)