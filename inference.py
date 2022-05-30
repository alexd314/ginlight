from argparse import ArgumentParser
import gin
import gin.torch
import torch
import pytorch_lightning as pl
from infrastructure.lightmain import LightMain
from infrastructure.utils.utils import get_class_constructor_params
import infrastructure.gin.gininfrastructure
import utils.gin

from absl import flags, app

def prepare_app():

    flags.DEFINE_multi_string('config',"config/main.gin","Main gin config")
    flags.DEFINE_multi_string('param',[],'Bind param to gin')

def run_main(used_args):
    FLAGS = flags.FLAGS
    gin.parse_config_files_and_bindings(FLAGS.config,FLAGS.param)
    main()

def get_lightmain_bound_params(lightmain : LightMain):

    params = gin.get_bindings('LightMain')
    return params

@gin.configurable
def main(lightmain : LightMain, trainer : pl.Trainer, chk_point_path : str, datamodule : pl.LightningDataModule):
    kwargs = get_lightmain_bound_params(lightmain)
    lightmain.load_from_checkpoint(chk_point_path,**kwargs)
    trainer.test(lightmain, datamodule = datamodule)

if __name__ == "__main__":
    prepare_app()
    app.run(run_main)
