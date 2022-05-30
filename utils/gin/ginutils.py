import gin
import utils

gin.external_configurable(utils.data.DataModule)
gin.external_configurable(utils.optim.StandardOptimizerFactory)
gin.external_configurable(utils.optim.MultipleOptimizerFactory)
gin.external_configurable(utils.model_init.XavierInitializer)
gin.external_configurable(utils.metrics.MetricProcessor)
gin.external_configurable(utils.logging.MetricFileLogger)
