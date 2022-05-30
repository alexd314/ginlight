import gin
import gin.torch
import pytorch_lightning as pl
import infrastructure.gin
import utils.gin
from infrastructure.lightmain import LightMain
from absl import flags, app

@gin.configurable
def main(lightmain : LightMain, trainer : pl.Trainer, datamodule : pl.LightningDataModule):
    trainer.fit(lightmain, datamodule = datamodule)

def prepare_app():

    flags.DEFINE_multi_string('config',"config/main.gin","Main gin config")
    flags.DEFINE_multi_string('param',[],'Bind param to gin')

def run_main(used_args):

    FLAGS = flags.FLAGS
    gin.parse_config_files_and_bindings(FLAGS.config,FLAGS.param)
    main()

if __name__ == "__main__":
    prepare_app()
    app.run(run_main)
