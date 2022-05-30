import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback

class XavierInitializer(Callback):

    def on_train_start(self, trainer, pl_module):

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

        for module in pl_module.modules():
            module.apply(init_weights)