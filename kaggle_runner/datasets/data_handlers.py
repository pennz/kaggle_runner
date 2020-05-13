import os
import pickle
from glob import glob

import numpy as np
import pandas as pd

from kaggle_runner.utils.kernel_utils import logger, rle2mask

# The following functions can be used to convert a value to a type compatible
# with tf.Example.


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
        # BytesList won't unpack a string from an EagerTensor.
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
            Y_train = np.zeros((self.im_height, self.im_width, 1), dtype=np.uint8)
            print("Getting train images and masks ... ")
            _id = path
            # sys.stdout.flush()
            # FIXME it cannot be put to autograph!!!
            raise RuntimeError("Pydicom read cannot be put to autograph!!!")
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
                                self.df.loc[_id_keystr, self.TARGET_COLUMN], 1024, 1024
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
        train_path = tf.keras.utils.get_file(TRAIN_URL.split("/")[-1], TRAIN_URL)
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
        _label_id_map = dict((name, index) for index, name in enumerate(SPECIES))
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
            ds = s.shuffle(buffer_size=int(len(labels) * (n_splits - 1) / n_splits))
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

        target_writer = tf.data.experimental.TFRecordWriter(f"target_{file_name}")
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

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        logger.debug("after transforming one feature to tf example proto")
        return example_proto.SerializeToString()

    @staticmethod
    def tf_serialize_example(f0, f1):
        print(PS_TF_DataHandler.serialize_PS_example(f0, f1))
        # the return type is
        # <a href="..../../versions/r2.0/api_docs/python/tf#string">
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
