import tensorflow as tf

from kaggle_datasets import KaggleDatasets
from kaggle_runner import logger
from kaggle_runner.defaults import DEBUG

AUTO = tf.data.experimental.AUTOTUNE

TPU_ADDRESS = os.environ.get('TPU_NAME')

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    BATCH_SIZE = 32 * strategy.num_replicas_in_sync
    logger.debug('Running on TPU: %s', tpu.master())
except ValueError as e:
    logger.error("%s",e)
    tpu = None
    strategy = tf.distribute.get_strategy()
    # strategy = None
    logger.debug('NOT running on TPU.')

    if DEBUG:
        BATCH_SIZE = 32 * 2
    else:
        BATCH_SIZE = 32 * 32

logger.info("REPLICAS: %d", strategy.num_replicas_in_sync)

GCS_DS_PATH = KaggleDatasets().get_gcs_path('jigsaw-multilingual-toxic-comment-classification')
GCS_DS_PATH_MINE = KaggleDatasets().get_gcs_path('jigsaw-multilingula-toxicity-token-encoded')

EPOCHS = 2
