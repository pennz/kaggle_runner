import os
import pickle
import re

import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from tokenizers import BertWordPieceTokenizer

from kaggle_runner import may_debug
from kaggle_runner.defaults import DEBUG
from kaggle_runner.utils.kernel_utils import (get_kaggle_dataset_input,
                                              get_obj_or_dump)
from kaggle_runner.utils.tpu import BATCH_SIZE, GCS_DS_PATH, strategy, tpu

#if DEBUG:
tf.executing_eagerly()
# Dataloading related
AUTO = tf.data.experimental.AUTOTUNE

# ### Load the training, validation, and testing datasets

DATA_PATH = "/kaggle/input/jigsaw-multilingual-toxic-comment-classification/"

if tpu is not None:
    DATA_PATH = GCS_DS_PATH + '/'
# os.listdir(DATA_PATH)

# +
TEST_PATH = DATA_PATH + "test.csv"
VAL_PATH = DATA_PATH + "validation.csv"
TRAIN_PATH = DATA_PATH + "jigsaw-toxic-comment-train.csv"

val_data = None
test_data = None
train_data = None

data_package = get_kaggle_dataset_input(
    "jigsaw-multilingula-toxicity-token-encoded/toxic_fast_tok_512.pk")
csv_data_package = get_kaggle_dataset_input(
    "jigsaw-multilingula-toxicity-token-encoded/toxic_csv.pk")

if csv_data_package is None:
    val_data = pd.read_csv(VAL_PATH)
    test_data = pd.read_csv(TEST_PATH)
    train_data = pd.read_csv(TRAIN_PATH)
    csv_data_package = get_obj_or_dump(
        "toxic_csv.pk", default=(val_data, test_data, train_data))
else:
    val_data, test_data, train_data = csv_data_package


if data_package is None:
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(
        'distilbert-base-multilingual-cased')

    save_path = '/kaggle/working/distilbert_base_uncased/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tokenizer.save_pretrained(save_path)

    def fast_encode(texts, tokenizer, chunk_size=240, maxlen=512):
        tokenizer.enable_truncation(max_length=maxlen)
        tokenizer.enable_padding(max_length=maxlen)
        all_ids = []

        for i in range(0, len(texts), chunk_size):
            text_chunk = texts[i:i+chunk_size].tolist()
            encs = tokenizer.encode_batch(text_chunk)
            all_ids.extend([enc.ids for enc in encs])

        return np.array(all_ids)

    fast_tokenizer = BertWordPieceTokenizer('distilbert_base_uncased/vocab.txt',
                                            lowercase=True)

# ### Clean the text (remove usernames and links)
# +
    val = val_data
    train = train_data

    def clean(text):
        text = text.fillna("fillna").str.lower()
        text = text.map(lambda x: re.sub('\\n', ' ', str(x)))
        text = text.map(lambda x: re.sub(
            "\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", '', str(x)))
        text = text.map(lambda x: re.sub(
            "\(http://.*?\s\(http://.*\)", '', str(x)))

        return text

    val["comment_text"] = clean(val["comment_text"])
    test_data["content"] = clean(test_data["content"])
    train["comment_text"] = clean(train["comment_text"])


# -

    x_train = fast_encode(train.comment_text.astype(str),
                          fast_tokenizer, maxlen=512)
    x_valid = fast_encode(val_data.comment_text.astype(str).values,
                          fast_tokenizer, maxlen=512)
    x_test = fast_encode(test_data.content.astype(str).values,
                         fast_tokenizer, maxlen=512)

# TODO just save it to disk or dataset for faster startup, and use it as a
# dataset
    y_valid = val.toxic.values  # no aux info for validation
# y_train = train.toxic.values  # TODO add aux data
### Define training, validation, and testing datasets

    y_train = np.stack([train.toxic.values, train.severe_toxic.values,
                        train.obscene.values, train.threat.values,
                        train.insult.values, train.identity_hate.values]).T

    data_package = get_obj_or_dump("toxic_fast_tok_512.pk", default=(x_train,
                                                                     y_train,
                                                                     x_valid,
                                                                     y_valid,
                                                                     x_test))
else:
    x_train, y_train, x_valid, y_valid, x_test = data_package

TRAIN_LEN = len(x_train)
VALID_LEN = len(x_valid)
TEST_LEN = len(x_test)

if DEBUG:
    TRAIN_LEN = TRAIN_LEN//10
    VALID_LEN = VALID_LEN//10
    x_train = x_train[:TRAIN_LEN, :140]
    y_train = y_train[:TRAIN_LEN]  # y just all pass, they are labels
    x_valid = x_valid[:VALID_LEN, :140]
    y_valid = y_valid[:VALID_LEN]  # it just one dimention
    # TEST_LEN = TEST_LEN//100
    # x_test  = x_test[:TEST_LEN, :140]


# +
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .batch(BATCH_SIZE)
)
