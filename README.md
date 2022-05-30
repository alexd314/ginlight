# ginlight
 A lightweight, tiny framework based on gin-config and pytorch-lightning, for training deep learning models oriented towards fast prototyping, minimal coding, and rapid experimentation.


# TODO

- [X] Setup dataset/dataloader for image classification
- [X] Support metric logging and measurement during training on validation data (on tensorboard)
- [X] Support metric logging at test time (on files)
- [X] Support logging of gin config/pipeline at train/test time
- [X] Support Optimizer Factory for creating/configuring optimizers
- [X] Support Fixed Random seed
- [X] Pipeline refactoring so that you don't need to give DataRegistry instance to compoents at config time
- [X] Pipeline refactoring to simplify value set/retrieval from DataRegistry
- [ ] Log optimizer settings to tensorboard / log files
