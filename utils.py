import datetime
import errno
import logging
import os
import pickle
import time
from collections import defaultdict, deque
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import torch
import torch.distributed as dist
import torch.utils.data
import torchvision
from IPython.core.debugger import set_trace
from PIL import Image, ImageFile
from tensorflow.keras import constraints, initializers, regularizers
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import math_ops
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


class AMQPURL:
    class AMQPURL_DEV:
        host = "termite.rmq.cloudamqp.com"  # (Load balanced)
        passwd = "QrBHPPxbsd8IuIxKrCnX3-RGoLKaFhYI"
        username = "drdsfaew"
        Vhost = "drdsfaew"
        # host = "127.0.0.1"  # (Load balanced)
        # passwd = "guest"
        # username = "guest"
        # Vhost = "/"

    def __init__(
        self,
        host=AMQPURL_DEV.host,
        passwd=AMQPURL_DEV.passwd,
        Vhost=AMQPURL_DEV.Vhost,
        username=AMQPURL_DEV.username,
    ):
        self.host = host
        self.passwd = passwd
        self.Vhost = Vhost
        self.username = username

    def string(self):
        Vhost = self.Vhost
        if self.Vhost == "/":
            Vhost = ""
        return f"amqp://{self.username}:{self.passwd}@{self.host}/{Vhost}"


BIN_FOLDER = (
    "/content/gdrivedata/My Drive/" if os.path.isdir(
        "/content/gdrivedata") else "./"
)


def get_logger():
    FORMAT = "[%(levelname)s]%(asctime)s:%(name)s:%(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    return logger


logger = get_logger()


def dump_obj(obj, filename, fullpath=False, force=False):
    if not fullpath:
        path = BIN_FOLDER + filename
    else:
        path = filename
    if not force and os.path.isfile(path):
        logger.debug(f"{path} already existed, not dumping")
    else:
        logger.debug(f"Overwrite {path}!")
        with open(path, "wb") as f:
            pickle.dump(obj, f)


def get_obj_or_dump(filename, fullpath=False, default=None):
    if not fullpath:
        path = BIN_FOLDER + filename
    else:
        path = filename

    if os.path.isfile(path):
        logger.debug("load " + filename)
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        if default is not None:
            logger.debug("dump :" + filename)
            dump_obj(default, filename)
        return default


def file_exist(filename, fullpath=False):
    if not fullpath:
        path = BIN_FOLDER + filename
    else:
        path = filename
    return os.path.isfile(path)


# 0.5 means no rebalance
def binary_crossentropy_with_focal_seasoned(
    y_true, logit_pred, beta=0.0, gamma=1.0, alpha=0.5, custom_weights_in_Y_true=True
):
    """
    :param alpha:weight for positive classes **loss**. default to 1- true
        positive cnts / all cnts, alpha range [0,1] for class 1 and 1-alpha
        for calss -1.   In practiceαmay be set by inverse class freqency or
        hyperparameter.
    :param custom_weights_in_Y_true:
    :return:
    """
    balanced = gamma * logit_pred + beta
    y_pred = math_ops.sigmoid(balanced)
    # only use gamma in this layer, easier to split out factor
    return binary_crossentropy_with_focal(
        y_true,
        y_pred,
        gamma=0,
        alpha=alpha,
        custom_weights_in_Y_true=custom_weights_in_Y_true,
    )


# 0.5 means no rebalance
def binary_crossentropy_with_focal(
    y_true, y_pred, gamma=1.0, alpha=0.5, custom_weights_in_Y_true=True
):
    """
    https://arxiv.org/pdf/1708.02002.pdf

    $$ FL(p_t) = -(1-p_t)^{\gamma}log(p_t) $$
    $$ p_t=p\: if\: y=1$$
    $$ p_t=1-p\: otherwise$$

    :param y_true:
    :param y_pred:
    :param gamma: make easier ones weights down
    :param alpha: weight for positive classes. default to 1 - (true positive cnts / all cnts),
        alpha range [0,1] for class 1 and 1-alpha for calss -1.   In practice
        α may be set by inverse class freqency or hyperparameter.
    :return: bce
    """
    # assert 0 <= alpha <= 1 and gamma >= 0
    # hyper parameters, just use the one for binary?
    # alpha = 1. # maybe smaller one can help, as multi-class will make the
    # error larger
    # gamma = 1.5 # for our problem, try different gamma

    # for binary_crossentropy, the implementation is in  tensorflow/tensorflow/python/keras/backend.py
    #       bce = target * alpha* (1-output+epsilon())**gamma * math_ops.log(output + epsilon())
    #       bce += (1 - target) *(1-alpha)* (output+epsilon())**gamma * math_ops.log(1 - output + epsilon())
    # return -bce # binary cross entropy
    eps = tf.keras.backend.epsilon()

    if custom_weights_in_Y_true:
        custom_weights = y_true[:, 1:2]
        y_true = y_true[:, :1]

    if 1.0 - eps <= gamma <= 1.0 + eps:
        bce = alpha * math_ops.multiply(
            1.0 - y_pred, math_ops.multiply(y_true, math_ops.log(y_pred + eps))
        )
        bce += (1 - alpha) * math_ops.multiply(
            y_pred, math_ops.multiply(
                (1.0 - y_true), math_ops.log(1.0 - y_pred + eps))
        )
    elif 0.0 - eps <= gamma <= 0.0 + eps:
        bce = alpha * math_ops.multiply(y_true, math_ops.log(y_pred + eps))
        bce += (1 - alpha) * math_ops.multiply(
            (1.0 - y_true), math_ops.log(1.0 - y_pred + eps)
        )
    else:
        gamma_tensor = tf.broadcast_to(
            tf.constant(gamma), tf.shape(input=y_pred))
        bce = alpha * math_ops.multiply(
            math_ops.pow(1.0 - y_pred, gamma_tensor),
            math_ops.multiply(y_true, math_ops.log(y_pred + eps)),
        )
        bce += (1 - alpha) * math_ops.multiply(
            math_ops.pow(y_pred, gamma_tensor),
            math_ops.multiply(
                (1.0 - y_true), math_ops.log(1.0 - y_pred + eps)),
        )

    if custom_weights_in_Y_true:
        return math_ops.multiply(-bce, custom_weights)
    else:
        return -bce


def reinitLayers(model):
    session = K.get_session()
    for layer in model.layers:
        # if isinstance(layer, keras.engine.topology.Container):
        if isinstance(layer, tf.keras.Model):
            reinitLayers(layer)
            continue
        print("LAYER::", layer.name)
        if layer.trainable is False:
            continue
        for v in layer.__dict__:
            v_arg = getattr(layer, v)
            if hasattr(v_arg, "initializer"):
                # not work for layer wrapper, like Bidirectional
                initializer_method = getattr(v_arg, "initializer")
                initializer_method.run(session=session)
                print("reinitializing layer {}.{}".format(layer.name, v))


class AttentionRaffel(Layer):
    def __init__(
        self,
        step_dim,
        W_regularizer=None,
        b_regularizer=None,
        W_constraint=None,
        b_constraint=None,
        bias=True,
        **kwargs,
    ):
        """
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        :param step_dim: feature vector length
        :param W_regularizer:
        :param b_regularizer:
        :param W_constraint:
        :param b_constraint:
        :param bias:
        :param kwargs:
        """
        super(AttentionRaffel, self).__init__(**kwargs)
        self.supports_masking = True
        self.init = "glorot_uniform"

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

    def get_config(self):
        config = {
            "step_dim": self.step_dim,
            "bias": self.bias,
            "W_regularizer": regularizers.serialize(self.W_regularizer),
            "b_regularizer": regularizers.serialize(self.b_regularizer),
            "W_constraint": constraints.serialize(self.W_constraint),
            "b_constraint": constraints.serialize(self.b_constraint),
        }
        base_config = super(AttentionRaffel, self).get_config()
        if "cell" in base_config:
            del base_config["cell"]
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        # Input shape 3D tensor with shape: `(samples, steps, features)`.
        # one step is means one bidirection?
        assert len(input_shape) == 3

        self.W = self.add_weight(
            "{}_W".format(self.name),
            (int(input_shape[-1]),),
            initializer=self.init,
            regularizer=self.W_regularizer,
            constraint=self.W_constraint,
        )
        self.features_dim = input_shape[-1]  # features dimention of input

        if self.bias:
            self.b = self.add_weight(
                "{}_b".format(self.name),
                (int(input_shape[1]),),
                initializer="zero",
                regularizer=self.b_regularizer,
                constraint=self.b_constraint,
            )
        else:
            self.b = None

        self.built = True

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # more like the alignment model, which scores how the inputs around
        # position j and the output at position i match
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(
            K.dot(
                K.reshape(x, (-1, features_dim)
                          ), K.reshape(self.W, (features_dim, 1))
            ),
            (-1, step_dim),
        )

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)  # activation

        # softmax
        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may
        # be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        # context vector c_i (or for this, only one c_i)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim


class NBatchProgBarLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(
        self,
        count_mode="samples",
        stateful_metrics=None,
        display_per_batches=1000,
        verbose=1,
        early_stop=False,
        patience_displays=0,
        epsilon=1e-7,
        batch_size=1024,
    ):
        super(NBatchProgBarLogger, self).__init__(count_mode, stateful_metrics)
        self.display_per_batches = 1 if display_per_batches < 1 else display_per_batches
        self.step_idx = 0  # across epochs
        self.display_idx = 0  # across epochs
        self.verbose = verbose

        # better way is subclass EearlyStopping callback.
        self.early_stop = early_stop
        self.patience_displays = patience_displays
        self.losses = np.empty(patience_displays, dtype=np.float32)
        self.losses_sum_display = 0
        self.epsilon = epsilon
        self.stopped_step = 0
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        self.epochs = self.params["epochs"]

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get("size", 0)
        # In case of distribution strategy we can potentially run multiple
        # steps at the same time, we should account for that in the `seen`
        # calculation.
        num_steps = logs.get("num_steps", 1)
        if self.use_steps:
            self.seen += num_steps
        else:
            self.seen += batch_size * num_steps

        for k in self.params["metrics"]:
            if k in logs:
                self.log_values.append((k, logs[k]))

        self.step_idx += 1
        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.early_stop:
            # only record for this batch, not the display. Should work
            loss = logs.get("loss")
            self.losses_sum_display += loss

        if self.step_idx % self.display_per_batches == 0:
            if self.verbose and self.seen < self.target:
                self.progbar.update(self.seen, self.log_values)

            if self.early_stop:
                avg_loss_per_display = (
                    self.losses_sum_display / self.display_per_batches
                )
                self.losses_sum_display = 0  # clear mannually...
                self.losses[
                    self.display_idx % self.patience_displays
                ] = avg_loss_per_display
                # but it still SGD, variance still, it just smaller by factor of
                # display_per_batches
                display_info_start_step = self.step_idx - self.display_per_batches + 1
                print(
                    f"\nmean: {avg_loss_per_display}, Step {display_info_start_step }({display_info_start_step*self.batch_size}) to {self.step_idx}({self.step_idx*self.batch_size}) for {self.display_idx}th display step"
                )

                self.display_idx += 1  # used in index, so +1 later
                if self.display_idx >= self.patience_displays:
                    std = np.std(
                        self.losses
                    )  # as SGD, always variance, so not a good way, need to learn from early stopping
                    std_start_step = (
                        self.step_idx
                        - self.display_per_batches * self.patience_displays
                        + 1
                    )
                    print(
                        f"mean: {np.mean(self.losses)}, std:{std} for Step {std_start_step}({std_start_step*self.batch_size}) to {self.step_idx}({self.step_idx*self.batch_size}) for {self.display_idx}th display steps"
                    )
                    if std < self.epsilon:
                        self.stopped_step = self.step_idx
                        self.model.stop_training = True
                        print(
                            f"Early Stop criterion met: std is {std} at Step {self.step_idx} for {self.display_idx}th display steps"
                        )

    def on_train_end(self, logs=None):
        if self.stopped_step > 0 and self.verbose > 0:
            print("Step %05d: early stopping" % (self.stopped_step + 1))


class PS_TF_DataHandler:
    def __init__(self):
        self.fns = None

    def to_tf_from_disk(self, fns, df, TARGET_COLUMN, im_height, im_width, im_chan):
        self.df = df
        self.TARGET_COLUMN = TARGET_COLUMN
        self.im_height = im_height
        self.im_width = im_width
        self.im_chan = im_chan

        fns_ds = tf.data.Dataset.from_tensor_slices(fns)
        image_ds = fns_ds.map(
            self.load_and_preprocess_image(imgPreprocessFlag=False),
            num_parallel_calls=2,
        )
        return image_ds

    def load_and_preprocess_image(self, imgPreprocessFlag=True):
        def _preprocess_image(img):
            raise NotImplementedError()

        # hard to do, as read_file, _id.split needs complicate op of tensor,
        # easier to first read numpy then save to tfrecord
        def _load_and_preprocess_image(path):
            X_train = np.zeros(
                (self.im_height, self.im_width, self.im_chan), dtype=np.uint8
            )
            Y_train = np.zeros(
                (self.im_height, self.im_width, 1), dtype=np.uint8)
            print("Getting train images and masks ... ")
            _id = path
            # sys.stdout.flush()
            dataset = pydicom.read_file(_id)
            _id_keystr = _id.split("/")[-1][:-4]
            X_train = np.expand_dims(dataset.pixel_array, axis=2)
            try:
                mask_data = self.df.loc[_id_keystr, self.TARGET_COLUMN]

                if "-1" in mask_data:
                    Y_train = np.zeros((1024, 1024, 1))
                else:
                    if type(mask_data) == str:
                        Y_train = np.expand_dims(
                            rle2mask(
                                self.df.loc[_id_keystr,
                                            self.TARGET_COLUMN], 1024, 1024
                            ),
                            axis=2,
                        )
                    else:
                        Y_train = np.zeros((1024, 1024, 1))
                        for x in mask_data:
                            Y_train = Y_train + np.expand_dims(
                                rle2mask(x, 1024, 1024), axis=2
                            )
            except KeyError:
                print(
                    f"Key {_id.split('/')[-1][:-4]} without mask, assuming healthy patient."
                )
                # Assume missing masks are empty masks.
                Y_train = np.zeros((1024, 1024, 1))

            if imgPreprocessFlag:
                return _preprocess_image(X_train), _preprocess_image(Y_train)
            return (X_train, Y_train)

        return _load_and_preprocess_image

    @staticmethod
    def maybe_download():
        # By default the file at the url origin is downloaded to the cache_dir
        # ~/.keras, placed in the cache_subdir datasets, and given the filename
        # fname
        train_path = tf.keras.utils.get_file(
            TRAIN_URL.split("/")[-1], TRAIN_URL)
        test_path = tf.keras.utils.get_file(TEST_URL.split("/")[-1], TEST_URL)

        return train_path, test_path

    @staticmethod
    def get_train_dataset(train_X_np, train_Y_np):  # join(dataset_dir,'labels.csv')
        image_ds = tf.data.Dataset.from_tensor_slices(train_X_np)
        image_mask_ds = tf.data.Dataset.from_tensor_slices(train_Y_np)

        return tf.data.Dataset.zip((image_ds, image_mask_ds))

    @staticmethod
    def load_data(train_path, test_path):
        """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
        # train_path, test_path = maybe_download()
        # here the test is really no lable we need to do CV in train part
        train_X = pickle.load(open(train_path, "rb"))  # (None, 2048)
        # (None, 2048) 2048 features from xception net
        to_predict_X = pickle.load(open(test_path, "rb"))

        try:
            labels = pd.read_csv(os.path.join(DATASET_DIR, "labels.csv"))
        except FileNotFoundError:
            labels = pd.read_csv(os.path.join(DATASET_DIR2, "labels.csv"))

        labels = labels["breed"].values.tolist()  # for all training data
        global SPECIES
        SPECIES = sorted(list(set(labels)))
        _label_id_map = dict((name, index)
                             for index, name in enumerate(SPECIES))
        train_y = [_label_id_map[label] for label in labels]

        return (train_X, train_y), to_predict_X

    @staticmethod
    def train_input_fn_bt(
        features,
        labels,
        batch_size,
        cv,
        cv_train=True,
        split_id=None,
        n_splits=None,
        ds=None,
        ds_len=-1,
    ):
        # for boost tree, need to prepare feature columns
        # 2048? columns, all float
        if cv:
            return PS_TF_DataHandler._input_fn_bt(
                features,
                labels,
                batch_size,
                shuffle=True,
                split_id=split_id,
                n_splits=n_splits,
                cv_train=cv_train,
                ds=ds,
                ds_len=ds_len,
            )
        else:
            return PS_TF_DataHandler._input_fn_bt(
                features, labels, batch_size, shuffle=True, cv=False, ds=ds
            )

    @staticmethod
    def eval_input_fn_bt(
        features, labels, batch_size, cv, split_id=None, n_splits=None
    ):
        if cv:
            return PS_TF_DataHandler._input_fn_bt(
                features,
                labels,
                batch_size,
                with_y=True,
                repeat=False,
                shuffle=False,
                split_id=split_id,
                n_splits=n_splits,
                cv_train=False,
            )
        else:
            return PS_TF_DataHandler._input_fn_bt(
                features,
                labels,
                batch_size,
                with_y=True,
                repeat=False,
                shuffle=False,
                cv=False,
            )

    @staticmethod
    def pred_input_fn_bt(features, batch_size):
        return PS_TF_DataHandler._input_fn_bt(
            features,
            None,
            batch_size,
            with_y=False,
            repeat=False,
            shuffle=False,
            cv=False,
        )

    @staticmethod
    # for these, we will need to extract all the points before:
    def _input_fn_bt(
        features,
        labels,
        batch_size,
        with_y=True,
        repeat=True,
        shuffle=True,
        split_id=-1,
        n_splits=10,
        cv=True,
        cv_train=True,
        ds=None,
        ds_len=-1,
    ):
        if ds is not None:
            if shuffle and ds_len <= 0:
                raise ValueError("shuffle need to now data length")
            data_len = ds_len
        else:
            data_len = len(labels)

            def _to_dict(f):
                # first to pandas data frame
                df = pd.DataFrame(
                    f, columns=[str(i) for i in range(features.shape[-1])]
                )
                return dict(df)

            features = _to_dict(features)
            if with_y:
                ds = tf.data.Dataset.from_tensor_slices((features, labels))
            else:
                ds = tf.data.Dataset.from_tensor_slices(features)

        if cv:
            assert split_id >= 0 and n_splits > 1 and split_id < n_splits
            if cv_train:
                ds = [ds.shard(n_splits, i) for i in range(n_splits)]

                shards_cross = [
                    ds[val_id] for val_id in range(n_splits) if val_id != split_id
                ]

                ds = shards_cross[0]
                for t in shards_cross[1:]:
                    ds = ds.concatenate(t)

                if shuffle:
                    # just memory is not enough ...
                    ds = ds.shuffle(
                        buffer_size=int(data_len * (n_splits - 1) / n_splits)
                    )
            else:  # cv part for evaluation, no need to shuffle
                ds = ds.shard(n_splits, split_id)
        else:
            if shuffle:
                ds = ds.shuffle(buffer_size=data_len)
        # after shuffle, we do cross validtation split

        # taken from Dan, https://stackoverflow.com/questions/39748660/how-to-perform-k-fold-cross-validation-with-tensorflow
        # will need to append id, then remove the id?
        # -> no need, we just split to 5 shards, then rearrange these shards
        if repeat and cv_train:
            ds = ds.repeat()
        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        # Return the dataset.
        return ds.batch(batch_size).prefetch(1)

    @staticmethod
    # for these, we will need to extract all the points before:
    def train_input_fn(features, labels, batch_size, split_id=-1, n_splits=10, cv=True):
        """An input function for training"""
        # read from the tfrecord file (save the extracted ones)(read the data)
        ds = tf.data.Dataset.from_tensor_slices((features, labels))
        if cv:
            assert split_id >= 0 and n_splits > 1 and split_id < n_splits
            ds = [ds.shard(n_splits, i) for i in range(n_splits)]

            shards_cross = [
                ds[val_id] for val_id in range(n_splits) if val_id != split_id
            ]

            s = shards_cross[0]
            for t in shards_cross[1:]:
                s = s.concatenate(t)

            # just memory is not enough ...
            ds = s.shuffle(buffer_size=int(
                len(labels) * (n_splits - 1) / n_splits))
        else:
            ds = ds.shuffle(buffer_size=len(labels))
        # after shuffle, we do cross validtation split

        # taken from Dan, https://stackoverflow.com/questions/39748660/how-to-perform-k-fold-cross-validation-with-tensorflow
        # will need to append id, then remove the id?
        # -> no need, we just split to 5 shards, then rearrange these shards
        ds = ds.repeat()
        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        # Return the dataset.
        return ds.batch(batch_size).prefetch(1)

    @staticmethod
    def eval_input_fn(features, labels, batch_size, split_id, n_splits=10):
        """An input function for evaluation or prediction"""
        assert split_id >= 0 and n_splits > 1 and split_id < n_splits
        if labels is None:
            # No labels, use only features.
            inputs = features
        else:
            inputs = (features, labels)

        # Convert the inputs to a Dataset.
        ds = tf.data.Dataset.from_tensor_slices(inputs)
        ds = ds.shard(n_splits, split_id)
        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        ds = ds.batch(batch_size)

        # Return the dataset.
        return ds

    @staticmethod
    def predict_input_fn(features, batch_size):
        """An input function for evaluation or prediction"""
        inputs = features

        # Convert the inputs to a Dataset.
        ds = tf.data.Dataset.from_tensor_slices(inputs)
        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        ds = ds.batch(batch_size)

        # Return the dataset.
        return ds

    @staticmethod
    # specific to data structure, need to split out later
    def to_tfrecord(ds, file_name="train_dev.tfrec"):
        ds = ds.map(lambda a, b: (tf.io.encode_jpeg(a), tf.io.encode_jpeg(b)))

        writer = tf.data.experimental.TFRecordWriter(file_name)
        writer.write(ds.map(lambda a, b: a))

        target_writer = tf.data.experimental.TFRecordWriter(
            f"target_{file_name}")
        target_writer.write(ds.map(lambda a, b: b))

        return

    @staticmethod
    def from_tfrecord():
        def _tf_read_jpeg(wc):
            pathes = sorted(glob(wc))
            logger.debug(f"recover data from {pathes}")

            ds = tf.data.TFRecordDataset(pathes)
            ds = ds.map(tf.io.decode_jpeg)
            return ds

        image_data_wildcard = "train_dev.*.tfrec"
        mask_data_wildcard = "target_train_dev.*.tfrec"
        return tf.data.Dataset.zip(
            (_tf_read_jpeg(image_data_wildcard), _tf_read_jpeg(mask_data_wildcard))
        )

    @staticmethod
    def serialize_PS_example(feature0, feature1):
        """
        NOT WORKING... don't know why
        Creates a tf.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the
        # tf.Example-compatible data type.
        assert feature0.shape[0] == 1 and feature0.shape[1] == 128
        assert (
            feature0.shape[0] == feature1.shape[0]
            and feature0.shape[1] == feature1.shape[1]
        )

        f0 = tf.reshape(feature0, [-1])
        f1 = tf.reshape(feature1, [-1])

        feature = {
            "image": _int64_feature_from_list(f0),
            "mask": _int64_feature_from_list(f1),
        }
        # Create a Features message using tf.train.Example.
        logger.debug("in transforming to tf example proto")

        example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
        logger.debug("after transforming one feature to tf example proto")
        return example_proto.SerializeToString()

    @staticmethod
    def tf_serialize_example(f0, f1):
        print(PS_TF_DataHandler.serialize_PS_example(f0, f1))
        # the return type is
        # <a href="../../../versions/r2.0/api_docs/python/tf#string">
        # <code>tf.string</code></a>.
        tf_string = tf.py_function(
            PS_TF_DataHandler.serialize_PS_example,
            (f0, f1),  # pass these args to the above function.
            tf.string,
        )
        return tf.reshape(tf_string, ())  # The result is a scalar

    @staticmethod
    def get_generator_with_features(features_dataset):
        def generator():
            for features in features_dataset:
                yield PS_TF_DataHandler.serialize_PS_example(*features)

        return generator


# The following functions can be used to convert a value to a type compatible
# with tf.Example.


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_from_list(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# Evaluation metric
# ref https://www.kaggle.com/jesperdramsch/intro-chest-xray-dicom-viz-u-nets-full-data
def dice_coef(y_true, y_pred, smooth=1, threshold=0.5):
    threshold = math_ops.cast(threshold, y_pred.dtype)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_pred_b = math_ops.cast(y_pred_f > threshold, y_pred.dtype)
    y_true_b = math_ops.cast(y_true_f > threshold, y_pred.dtype)

    intersection = K.sum(y_true_b * y_pred_b)
    return (2.0 * intersection + smooth) / (K.sum(y_true_b) + K.sum(y_pred_b) + smooth)


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position: current_position + lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def clear(self):
        self.total = 0.0
        self.count = 0
        self.deque.clear()

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        if len(self.deque) == 0:
            return "_NULL_"
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty(
            (max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all
    processes have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t", log_file_name="metric.log"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.log_file = open(log_file_name, "a", buffering=1)

    def __del__(self):
        self.log_file.close()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def clear(self):
        for meter in self.meters.values():
            if meter is not None:
                meter.clear()

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(
                type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 1
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = self.delimiter.join(
            [
                header,
                "[{0" + space_fmt + "}/{1}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
                "max mem: {memory:.0f}",
            ]
        )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            # print the first to let you know it is running....
            if i % (print_freq) == 0 or i == len(iterable):
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                self.print_and_log_to_file(
                    log_msg.format(
                        i,
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB,
                    )
                )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.print_and_log_to_file(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )

    def print_and_log_to_file(self, s):
        print(s)
        self.log_file.write(s + "\n")


def collate_fn(batch):
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def mask_to_rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 1:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1
    return " " + " ".join(rle)


class EarlyStopping(object):
    """EarlyStop for pytorch
    refer to
    https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d"""

    def __init__(self, mode="min", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - \
                    (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + \
                    (best * min_delta / 100)


def online_mean_and_sd(loader, data_map=None):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    credit xwkuang5
    @https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/7
    """
    cnt = 0
    # fst_moment = torch.empty(3)
    # snd_moment = torch.empty(3)
    fst_moment = np.zeros(3)
    snd_moment = np.zeros(3)

    for data in loader:
        if data_map is not None:
            data = data_map(data)
        data = np.array([t.numpy() for t in data])
        b, c, h, w = data.shape  # data here is tuple... if loader batch > 1
        nb_pixels = b * h * w
        # sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_ = data.sum(axis=0).sum(axis=-1).sum(axis=-1)
        # sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        sum_of_square = (data ** 2).sum(axis=0).sum(axis=-1).sum(axis=-1)
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, np.sqrt(snd_moment - fst_moment ** 2)


# dataset = MyDataset() loader = DataLoader( dataset, batch_size=1,
# num_workers=1, shuffle=False)
#
# mean, std = online_mean_and_sd(loader)
