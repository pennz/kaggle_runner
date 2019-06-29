import os
import pickle
import logging
from glob import glob

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
import os
import pandas as pd
import pydicom

from IPython.core.debugger import set_trace

#BIN_FOLDER = './'
#if os.path.isdir('/content/gdrivedata'):
BIN_FOLDER = '/content/gdrivedata/My Drive/' if os.path.isdir('/content/gdrivedata') else './'

def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger

logger = get_logger()

def dump_obj(obj, filename, fullpath=False, force=False):
    if not fullpath:
        path = BIN_FOLDER+filename
    else:
        path = filename
    if not force and os.path.isfile(path):
        logger.debug(f"{path} already existed, not dumping")
    else:
        logger.debug(f"Overwrite {path}!")
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

def get_obj_or_dump(filename, fullpath=False, default=None):
    if not fullpath:
        path = BIN_FOLDER+filename
    else:
        path = filename

    if os.path.isfile(path):
        logger.debug("load "+filename)
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        if default is not None:
            logger.debug("dump :"+filename)
            dump_obj(default, filename)
        return default

def file_exist(filename, fullpath=False):
    if not fullpath:
        path = BIN_FOLDER+filename
    else:
        path = filename
    return os.path.isfile(path)

def binary_crossentropy_with_focal_seasoned(y_true, logit_pred, beta=0., gamma=1., alpha=0.5, custom_weights_in_Y_true=True):  # 0.5 means no rebalance
    """
    :param alpha:weight for positive classes **loss**. default to 1- true positive cnts / all cnts, alpha range [0,1] for class 1 and 1-alpha
        for calss -1.   In practiceαmay be set by inverse class freqency or hyperparameter.
    :param custom_weights_in_Y_true:
    :return:
    """
    balanced = gamma*logit_pred + beta
    y_pred = math_ops.sigmoid(balanced)
    return binary_crossentropy_with_focal(y_true, y_pred, gamma=0, alpha=alpha, custom_weights_in_Y_true=custom_weights_in_Y_true)  # only use gamma in this layer, easier to split out factor


def binary_crossentropy_with_focal(y_true, y_pred, gamma=1., alpha=0.5, custom_weights_in_Y_true=True):  # 0.5 means no rebalance
    """
    https://arxiv.org/pdf/1708.02002.pdf

    $$ FL(p_t) = -(1-p_t)^{\gamma}log(p_t) $$
    $$ p_t=p\: if\: y=1$$
    $$ p_t=1-p\: otherwise$$

    :param y_true:
    :param y_pred:
    :param gamma: make easier ones weights down
    :param alpha: weight for positive classes. default to 1- true positive cnts / all cnts, alpha range [0,1] for class 1 and 1-alpha
        for calss -1.   In practiceαmay be set by inverse class freqency or hyperparameter.
    :return:
    """
    # assert 0 <= alpha <= 1 and gamma >= 0
    # hyper parameters, just use the one for binary?
    # alpha = 1. # maybe smaller one can help, as multi-class will make the error larger
    # gamma = 1.5 # for our problem, try different gamma

    # for binary_crossentropy, the implementation is in  tensorflow/tensorflow/python/keras/backend.py
    #       bce = target * alpha* (1-output+epsilon())**gamma * math_ops.log(output + epsilon())
    #       bce += (1 - target) *(1-alpha)* (output+epsilon())**gamma * math_ops.log(1 - output + epsilon())
    # return -bce # binary cross entropy
    eps = tf.keras.backend.epsilon()

    if custom_weights_in_Y_true:
        custom_weights = y_true[:, 1:2]
        y_true = y_true[:, :1]

    if 1. - eps <= gamma <= 1. + eps:
        bce = alpha * math_ops.multiply(1. - y_pred, math_ops.multiply(y_true, math_ops.log(y_pred + eps)))
        bce += (1 - alpha) * math_ops.multiply(y_pred,
                                               math_ops.multiply((1. - y_true), math_ops.log(1. - y_pred + eps)))
    elif 0. - eps <= gamma <= 0. + eps:
        bce = alpha * math_ops.multiply(y_true, math_ops.log(y_pred + eps))
        bce += (1 - alpha) * math_ops.multiply((1. - y_true), math_ops.log(1. - y_pred + eps))
    else:
        gamma_tensor = tf.broadcast_to(tf.constant(gamma), tf.shape(input=y_pred))
        bce = alpha * math_ops.multiply(math_ops.pow(1. - y_pred, gamma_tensor),
                                        math_ops.multiply(y_true, math_ops.log(y_pred + eps)))
        bce += (1 - alpha) * math_ops.multiply(math_ops.pow(y_pred, gamma_tensor),
                                               math_ops.multiply((1. - y_true), math_ops.log(1. - y_pred + eps)))

    if custom_weights_in_Y_true:
        return math_ops.multiply(-bce, custom_weights)
    else:
        return -bce

def reinitLayers(model):
    session = K.get_session()
    for layer in model.layers:
        #if isinstance(layer, keras.engine.topology.Container):
        if isinstance(layer, tf.keras.Model):
            reinitLayers(layer)
            continue
        print("LAYER::", layer.name)
        if layer.trainable == False:
            continue
        for v in layer.__dict__:
            v_arg = getattr(layer, v)
            if hasattr(v_arg,'initializer'):  # not work for layer wrapper, like Bidirectional
                initializer_method = getattr(v_arg, 'initializer')
                initializer_method.run(session=session)
                print('reinitializing layer {}.{}'.format(layer.name, v))

from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints
import tensorflow.keras.backend as K

class AttentionRaffel(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
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
        self.init = 'glorot_uniform'

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

    def get_config(self):
        config = {
            'step_dim':
                self.step_dim,
            'bias':
                self.bias,
            'W_regularizer':
                regularizers.serialize(self.W_regularizer),
            'b_regularizer':
                regularizers.serialize(self.b_regularizer),
            'W_constraint':
                constraints.serialize(self.W_constraint),
            'b_constraint':
                constraints.serialize(self.b_constraint),
        }
        base_config = super(AttentionRaffel, self).get_config()
        if 'cell' in base_config: del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        # Input shape 3D tensor with shape: `(samples, steps, features)`.
        # one step is means one bidirection?
        assert len(input_shape) == 3

        self.W = self.add_weight('{}_W'.format(self.name),
                                 (int(input_shape[-1]),),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]  # features dimention of input

        if self.bias:
            self.b = self.add_weight('{}_b'.format(self.name),
                                     (int(input_shape[1]),),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # more like the alignment model, which scores how the inputs around position j and the output
        # at position i match
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)  # activation

        # softmax
        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a  # context vector c_i (or for this, only one c_i)
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim

class NBatchProgBarLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, count_mode='samples', stateful_metrics=None, display_per_batches=1000, verbose=1,
                 early_stop=False, patience_displays=0, epsilon=1e-7, batch_size=1024):
        super(NBatchProgBarLogger, self).__init__(count_mode, stateful_metrics)
        self.display_per_batches = 1 if display_per_batches < 1 else display_per_batches
        self.step_idx = 0  # across epochs
        self.display_idx = 0  # across epochs
        self.verbose = verbose

        self.early_stop = early_stop  # better way is subclass EearlyStopping callback.
        self.patience_displays = patience_displays
        self.losses = np.empty(patience_displays, dtype=np.float32)
        self.losses_sum_display = 0
        self.epsilon = epsilon
        self.stopped_step = 0
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        # In case of distribution strategy we can potentially run multiple steps
        # at the same time, we should account for that in the `seen` calculation.
        num_steps = logs.get('num_steps', 1)
        if self.use_steps:
            self.seen += num_steps
        else:
            self.seen += batch_size * num_steps

        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        self.step_idx += 1
        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.early_stop:
            loss = logs.get('loss')  # only record for this batch, not the display. Should work
            self.losses_sum_display += loss

        if self.step_idx % self.display_per_batches == 0:
            if self.verbose and self.seen < self.target:
                self.progbar.update(self.seen, self.log_values)

            if self.early_stop:
                avg_loss_per_display = self.losses_sum_display / self.display_per_batches
                self.losses_sum_display = 0  # clear mannually...
                self.losses[self.display_idx % self.patience_displays] = avg_loss_per_display
                # but it still SGD, variance still, it just smaller by factor of display_per_batches
                display_info_start_step = self.step_idx - self.display_per_batches + 1
                print(
                    f'\nmean: {avg_loss_per_display}, Step {display_info_start_step }({display_info_start_step*self.batch_size}) to {self.step_idx}({self.step_idx*self.batch_size}) for {self.display_idx}th display step')

                self.display_idx += 1  # used in index, so +1 later
                if self.display_idx >= self.patience_displays:
                    std = np.std(
                        self.losses)  # as SGD, always variance, so not a good way, need to learn from early stopping
                    std_start_step = self.step_idx - self.display_per_batches * self.patience_displays + 1
                    print(f'mean: {np.mean(self.losses)}, std:{std} for Step {std_start_step}({std_start_step*self.batch_size}) to {self.step_idx}({self.step_idx*self.batch_size}) for {self.display_idx}th display steps')
                    if std < self.epsilon:
                        self.stopped_step = self.step_idx
                        self.model.stop_training = True
                        print(
                            f'Early Stop criterion met: std is {std} at Step {self.step_idx} for {self.display_idx}th display steps')

    def on_train_end(self, logs=None):
        if self.stopped_step > 0 and self.verbose > 0:
            print('Step %05d: early stopping' % (self.stopped_step + 1))

from enum import Enum

class KernelRunningState(Enum):
    INIT_DONE = 1
    PREPARE_DATA_DONE = 2
    TRAINING_DONE = 3
    EVL_DEV_DONE = 4
    SAVE_SUBMISSION_DONE = 5

class PS_TF_DataHandler():
    def __init__(self):
        self.fns = None

    def to_tf_from_disk(self, fns, df, TARGET_COLUMN, im_height, im_width, im_chan):
        self.df = df
        self.TARGET_COLUMN = TARGET_COLUMN
        self.im_height = im_height
        self.im_width = im_width
        self.im_chan = im_chan

        fns_ds = tf.data.Dataset.from_tensor_slices(fns)
        image_ds = fns_ds.map(self.load_and_preprocess_image(imgPreprocessFlag=False), num_parallel_calls=2)
        return image_ds

    def load_and_preprocess_image(self, imgPreprocessFlag=True):
        def _preprocess_image(img):
            raise NotImplementedError()

        def _load_and_preprocess_image(path):  # hard to do, as read_file, _id.split needs complicate op of tensor, easier to first read numpy then save to tfrecord
            X_train = np.zeros((self.im_height, self.im_width, self.im_chan), dtype=np.uint8)
            Y_train = np.zeros((self.im_height, self.im_width, 1), dtype=np.uint8)
            print('Getting train images and masks ... ')
            _id = path
            #sys.stdout.flush()
            dataset = pydicom.read_file(_id)
            _id_keystr = _id.split('/')[-1][:-4]
            X_train = np.expand_dims(dataset.pixel_array, axis=2)
            try:
                mask_data = self.df.loc[_id_keystr, self.TARGET_COLUMN]

                if '-1' in mask_data:
                    Y_train = np.zeros((1024, 1024, 1))
                else:
                    if type(mask_data) == str:
                        Y_train = np.expand_dims(
                            rle2mask(self.df.loc[_id_keystr, self.TARGET_COLUMN], 1024, 1024), axis=2)
                    else:
                        Y_train = np.zeros((1024, 1024, 1))
                        for x in mask_data:
                            Y_train = Y_train + np.expand_dims(rle2mask(x, 1024, 1024), axis=2)
            except KeyError:
                print(f"Key {_id.split('/')[-1][:-4]} without mask, assuming healthy patient.")
                Y_train = np.zeros((1024, 1024, 1))  # Assume missing masks are empty masks.

            if imgPreprocessFlag: return _preprocess_image(X_train),_preprocess_image(Y_train)
            return (X_train,Y_train)
        return _load_and_preprocess_image

    @staticmethod
    def maybe_download():
        # By default the file at the url origin is downloaded to the cache_dir ~/.keras,
        # placed in the cache_subdir datasets, and given the filename fname
        train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
        test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

        return train_path, test_path

    @staticmethod
    def get_train_dataset(train_X_np,train_Y_np): #join(dataset_dir,'labels.csv')
        image_ds = tf.data.Dataset.from_tensor_slices(train_X_np)
        image_mask_ds = tf.data.Dataset.from_tensor_slices(train_Y_np)

        return tf.data.Dataset.zip((image_ds, image_mask_ds))

    @staticmethod
    def load_data(train_path, test_path):
        """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
        #train_path, test_path = maybe_download() # here the test is really no lable
        # we need to do CV in train part
        train_X = pickle.load(open(train_path,"rb")) #(None, 2048)
        to_predict_X = pickle.load(open(test_path,"rb")) #(None, 2048) 2048 features from xception net

        try:
            labels = pd.read_csv(os.path.join(DATASET_DIR, 'labels.csv'))
        except FileNotFoundError:
            labels = pd.read_csv(os.path.join(DATASET_DIR2, 'labels.csv'))

        labels = labels['breed'].values.tolist() # for all training data
        global SPECIES
        SPECIES = sorted(list(set(labels)))
        _label_id_map = dict((name, index) for index,name in enumerate(SPECIES))
        train_y = [_label_id_map[label] for label in labels]

        return (train_X, train_y), to_predict_X

    @staticmethod
    def train_input_fn_bt(features, labels, batch_size, cv, cv_train=True, split_id=None, n_splits=None, ds=None, ds_len=-1):
        # for boost tree, need to prepare feature columns
        # 2048? columns, all float
        if cv:
            return PS_TF_DataHandler._input_fn_bt(features, labels, batch_size, shuffle=True, split_id=split_id, n_splits=n_splits,cv_train=cv_train, ds=ds, ds_len=ds_len)
        else:
            return PS_TF_DataHandler._input_fn_bt(features, labels, batch_size, shuffle=True, cv=False, ds=ds)

    @staticmethod
    def eval_input_fn_bt(features, labels, batch_size, cv, split_id=None, n_splits=None):
        if cv:
            return PS_TF_DataHandler._input_fn_bt(features, labels, batch_size,with_y=True, repeat=False, shuffle=False, split_id=split_id, n_splits=n_splits,cv_train=False)
        else:
            return PS_TF_DataHandler._input_fn_bt(features, labels, batch_size,with_y=True, repeat=False, shuffle=False, cv=False)

    @staticmethod
    def pred_input_fn_bt(features, batch_size):
        return PS_TF_DataHandler._input_fn_bt(features, None, batch_size, with_y=False, repeat=False, shuffle=False, cv=False)

    @staticmethod
    def _input_fn_bt(features, labels, batch_size, with_y=True,repeat=True, shuffle=True, split_id=-1, n_splits=10, cv=True, cv_train=True, ds=None, ds_len=-1): # for these, we will need to extract all the points before:
        if ds is not None:
            if shuffle and ds_len <= 0:
                raise ValueError('shuffle need to now data length')
            data_len = ds_len
        else:
            data_len = len(labels)
            def _to_dict(f):
                # first to pandas data frame
                df = pd.DataFrame(f, columns=[str(i) for i in range(features.shape[-1])])
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

                shards_cross = [ds[val_id] for val_id in range(n_splits) if val_id != split_id]

                ds = shards_cross[0]
                for t in shards_cross[1:]:
                    ds = ds.concatenate(t)

                if shuffle:
                    ds = ds.shuffle(buffer_size=int(data_len*(n_splits-1)/n_splits)) # just memory is not enough ...
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
    def train_input_fn(features, labels, batch_size,split_id=-1, n_splits=10, cv=True): # for these, we will need to extract all the points before:
        """An input function for training"""
        # read from the tfrecord file (save the extracted ones)(read the data)
        ds = tf.data.Dataset.from_tensor_slices((features, labels))
        if cv:
            assert split_id >= 0 and n_splits > 1 and split_id < n_splits
            ds = [ds.shard(n_splits, i) for i in range(n_splits)]

            shards_cross = [ds[val_id] for val_id in range(n_splits) if val_id != split_id]

            s = shards_cross[0]
            for t in shards_cross[1:]:
                s = s.concatenate(t)

            ds = s.shuffle(buffer_size=int(len(labels)*(n_splits-1)/n_splits)) # just memory is not enough ...
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
    def to_tfrecord(ds, file_name='train_dev.tfrec'):  # specific to data structure, need to split out later
        # self.dev_ds.take(1)
        # <TakeDataset shapes: ((None, 1024, 1024, 1), (None, 1024, 1024, 1)), types: (tf.uint8, tf.uint8)>
        #ds = ds.map(lambda a, b: (tf.io.encode_jpeg(tf.reshape(a, a.shape[1:])), tf.io.encode_jpeg(tf.reshape(b, b.shape[1:])) ) )
        #ds = ds.map(lambda a, b: (tf.image.encode_png(a), tf.image.encode_png(b) ) )
        ds = ds.map(lambda a, b: (tf.io.encode_jpeg(a), tf.io.encode_jpeg(b) ) )

        writer = tf.data.experimental.TFRecordWriter(file_name)
        #writer.write(tf.data.Dataset.zip(a_jpeg,b_jpeg))
        writer.write(ds.map(lambda a, b: a))

        target_writer = tf.data.experimental.TFRecordWriter(f'target_{file_name}')
        target_writer.write(ds.map(lambda a, b: b))

        return

    @staticmethod
    def from_tfrecord():
        def _tf_read_jpeg(wc):
            pathes = sorted(glob(wc))
            logger.debug(f'recover data from {pathes}')

            ds = tf.data.TFRecordDataset(pathes)
            ds = ds.map(tf.io.decode_jpeg)
            return ds
        image_data_wildcard = 'train_dev.*.tfrec'
        mask_data_wildcard = 'target_train_dev.*.tfrec'
        return tf.data.Dataset.zip((_tf_read_jpeg(image_data_wildcard), _tf_read_jpeg(mask_data_wildcard)))

    @staticmethod
    def serialize_PS_example(feature0, feature1):
        """
        NOT WORKING... don't know why
        Creates a tf.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        assert feature0.shape[0] == 1 and feature0.shape[1] == 128
        assert feature0.shape[0] == feature1.shape[0] and feature0.shape[1] == feature1.shape[1]

        f0 = tf.reshape(feature0, [-1])
        f1 = tf.reshape(feature1, [-1])

        feature = {
            'image': _int64_feature_from_list(f0),
            'mask': _int64_feature_from_list(f1),
        }
        # Create a Features message using tf.train.Example.
        logger.debug('in transforming to tf example proto')

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        logger.debug('after transforming one feature to tf example proto')
        return example_proto.SerializeToString()

    @staticmethod
    def tf_serialize_example(f0, f1):
        print(PS_TF_DataHandler.serialize_PS_example(f0, f1))
        tf_string = tf.py_function(
            PS_TF_DataHandler.serialize_PS_example,
            (f0, f1),  # pass these args to the above function.
            tf.string)  # the return type is <a href="../../../versions/r2.0/api_docs/python/tf#string"><code>tf.string</code></a>.
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
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
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

class KaggleKernel:
    def __init__(self):
        self.model = None
        self.model_metrics = []
        self.model_loss = None

        self.train_X = None
        self.train_Y = None

        self.dev_X = None
        self.dev_Y = None

        self.test_X = None

        self.result_analyzer = None  # for analyze the result

        self._stage = KernelRunningState.INIT_DONE

    def build_and_set_model(self):
        pass

    def train_model(self):
        pass

    def set_model(self):
        pass

    def set_loss(self):
        pass

    def set_metrics(self):
        """
        set_metrics for model training

        :return: None
        """
        pass

    def set_result_analyzer(self):
        pass

    def pre_prepare_data_hook(self):
        pass

    def after_prepare_data_hook(self):
        pass

    def prepare_train_data(self):
        pass

    def prepare_dev_data(self):
        pass

    def prepare_test_data(self):
        pass

    def save_result(self):
        pass

    def dump_state(self, exec_flag=False):
        logger.debug(f'state {self._stage}')
        if exec_flag:
            logger.debug(f'dumping state {self._stage}')
            dump_obj(self, f'run_state_{self._stage}.pkl')
            #dump_obj(self, 'run_state.pkl', force=True)  # too large

    def run(self, start_stage=None, end_stage=KernelRunningState.SAVE_SUBMISSION_DONE, dump_flag=False):
        """

        :param start_stage: if set, will overwrite the stage
        :param end_stage:
        :param dump_flag:
        :return:
        """
        self.continue_run(start_stage=start_stage, end_stage=end_stage, dump_flag=dump_flag)

    def continue_run(self, start_stage=None, end_stage=KernelRunningState.SAVE_SUBMISSION_DONE, dump_flag=False):
        if start_stage is not None:
            assert start_stage.value < end_stage.value
            self._stage = start_stage

        if self._stage.value < KernelRunningState.PREPARE_DATA_DONE.value:
            self.pre_prepare_data_hook()
            self.prepare_train_data()
            self.prepare_dev_data()
            self.prepare_test_data()
            self.after_prepare_data_hook()

            self._stage = KernelRunningState.PREPARE_DATA_DONE
            self.dump_state(exec_flag=dump_flag)
            if self._stage.value >= end_stage.value: return

        if self._stage.value < KernelRunningState.TRAINING_DONE.value:
            self.pre_train()
            self.build_and_set_model()
            self.train_model()
            self.after_train()

            self._stage = KernelRunningState.TRAINING_DONE
            self.dump_state(exec_flag=dump_flag)
            if self._stage.value >= end_stage.value: return

        if self._stage.value < KernelRunningState.EVL_DEV_DONE.value:
            self.set_result_analyzer()

            self._stage = KernelRunningState.EVL_DEV_DONE
            self.dump_state(exec_flag=dump_flag)
            if self._stage.value >= end_stage.value: return

        if self._stage.value < KernelRunningState.SAVE_SUBMISSION_DONE.value:
            self.save_result()

            self._stage = KernelRunningState.SAVE_SUBMISSION_DONE
            self.dump_state(exec_flag=dump_flag)
            if self._stage.value >= end_stage.value: return

    @classmethod
    def _load_state(cls, stage=None, file_name='run_state.pkl'):
        """

        :param file_name:
        :return: the kernel object, need to continue
        """
        if stage is not None:
            file_name = f'run_state_{stage}.pkl'
        logger.debug(f'restore from {file_name}')
        return get_obj_or_dump(filename=file_name)

    @classmethod
    def load_state_continue_run(cls, file_name='run_state.pkl'):
        """

        :param file_name:
        :return: the kernel object, need to continue
        """
        self = cls._load_state(file_name=file_name)
        self.continue_run()

    def pre_train(self):
        pass

    def after_train(self):
        pass


#Evaluation metric
#ref https://www.kaggle.com/jesperdramsch/intro-chest-xray-dicom-viz-u-nets-full-data
def dice_coef(y_true, y_pred, smooth=1, threshold=0.5):
    threshold = math_ops.cast(threshold, y_pred.dtype)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_pred_b = math_ops.cast(y_pred_f > threshold, y_pred.dtype)
    y_true_b = math_ops.cast(y_true_f > threshold, y_pred.dtype)

    intersection = K.sum(y_true_b * y_pred_b)
    return (2. * intersection + smooth) / (K.sum(y_true_b) + K.sum(y_pred_b) + smooth)

def rle2mask(rle, width, height):
    mask= np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 1
        current_position += lengths[index]

