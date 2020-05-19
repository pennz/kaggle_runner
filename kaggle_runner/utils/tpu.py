import tensorflow as tf

from kaggle_datasets import KaggleDatasets
from kaggle_runner import logger
from kaggle_runner.defaults import DEBUG

AUTO = tf.data.experimental.AUTOTUNE

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    BATCH_SIZE = 32 * strategy.num_replicas_in_sync
except ValueError as e:
    logger.error("%s",e)
    tpu = None
    strategy = None

    if DEBUG:
        BATCH_SIZE = 32 * 2
    else:
        BATCH_SIZE = 32 * 32

GCS_DS_PATH = KaggleDatasets().get_gcs_path('jigsaw-multilingual-toxic-comment-classification')

EPOCHS = 2
