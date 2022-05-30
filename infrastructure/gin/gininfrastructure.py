from datetime import datetime
import gin
import infrastructure.lightmain
import infrastructure.logging
#import infrastructure.lightlogging
import infrastructure.pipeline
import infrastructure.registry
import utils.metrics
import typing as tp

# infrastructure configurables
gin.external_configurable(dict)
gin.external_configurable(infrastructure.lightmain.LightMain)
gin.external_configurable(infrastructure.registry.DataRegistryCleaner)
# gin.external_configurable(infrastructure.logging.PipelineHyperparamLogger)
# gin.external_configurable(infrastructure.logging.AttributeLogger)
gin.external_configurable(infrastructure.logging.config.ObjectTreeLogger)
gin.external_configurable(infrastructure.logging.lightning.ScalarLogger)
gin.external_configurable(infrastructure.logging.lightning.TensorboardImageLogger)
gin.external_configurable(infrastructure.logging.lightning.TensorboardImagePairLogger)
# gin.external_configurable(infrastructure.lightlogging.AverageTensorLogger)
# gin.external_configurable(infrastructure.lightlogging.DictLogger)
gin.external_configurable(infrastructure.logging.config.GinConfigLogger)
gin.external_configurable(infrastructure.pipeline.Pipeline)
gin.external_configurable(infrastructure.pipeline.Component)
gin.external_configurable(utils.metrics.MetricProcessor)
#######################################

########################################################################
@gin.configurable
def generate_log_dir(dir_prefix: str, dir_suffix : str):
    now = datetime.now()
    ts = now.strftime("%d-%m-%Y-%H-%M-%S")
    dirname = dir_prefix + ts + dir_suffix
    return dirname

@gin.configurable
def strcat(s : tp.List[str]) -> str:
    return "".join(s)

