import os
import tensorflow as tf

from kaggle_runner import logger, may_debug
from kaggle_runner.defaults import DEBUG

AUTO = tf.data.experimental.AUTOTUNE

may_debug()
try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(os.getenv("TPU_NAME"))
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    strategy = tf.distribute.experimental.TPUStrategy(tpu_resolver)
    BATCH_SIZE = 32 * strategy.num_replicas_in_sync
    logger.debug('Running on TPU: %s', tpu_resolver.master())
except ValueError as e:
    logger.error("%s",e)
    tpu_resolver = None
    strategy = tf.distribute.get_strategy()
    # strategy = None
    logger.debug('NOT running on TPU.')

    if DEBUG:
        BATCH_SIZE = 32 * 2
    else:
        BATCH_SIZE = 32 * 32

logger.info("REPLICAS: %d", strategy.num_replicas_in_sync)


EPOCHS = 2
