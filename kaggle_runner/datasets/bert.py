import os
import pickle
import re

import numpy as np
import pandas as pd
import tensorflow as tf

from kaggle_runner import may_debug, logger
from kaggle_runner.defaults import DEBUG
from kaggle_runner.utils.kernel_utils import (get_kaggle_dataset_input,
                                              get_obj_or_dump)
#from kaggle_runner.utils.tpu import (strategy, tpu_resolver)
from kaggle_runner.hub.bert.extract_features import load_data, get_tokenizer


TOKENS_PATH = "/kaggle/input/jigsaw-toxic-token-ids-for-bert"
PRETRAIND_PICKLE_AND_MORE='/kaggle/input/toxic-multilang-trained-torch-model'

strategy=None
tpu_resolver=None
BATCH_SIZE = 32 * 2

if tpu_resolver is None:
    if DEBUG:
        BATCH_SIZE = 32 * 2
    else:
        BATCH_SIZE = 32 * 32
elif strategy is not None:
    BATCH_SIZE = 32 * strategy.num_replicas_in_sync

if tpu_resolver is None:
    DATA_PATH = "/kaggle/input/jigsaw-multilingual-toxic-comment-classification/"
    BERT_BASE_DIR = "/kaggle/input/bert-pretrained-models" + \
        '/multi_cased_L-12_H-768_A-12' + '/multi_cased_L-12_H-768_A-12'
else:
    from kaggle_datasets import KaggleDatasets
    GCS_DS_PATH = KaggleDatasets().get_gcs_path(
        'jigsaw-multilingual-toxic-comment-classification')
    GCS_BERT_PRETRAINED = KaggleDatasets().get_gcs_path('bert-pretrained-models') + \
        '/multi_cased_L-12_H-768_A-12'+'/multi_cased_L-12_H-768_A-12'

    DATA_PATH = GCS_DS_PATH + '/'
    BERT_BASE_DIR = GCS_BERT_PRETRAINED


def pickle_data(max_seq_length=128, bert_base_dir=BERT_BASE_DIR, output="features.pkl"):
    # --vocab_file="$BERT_BASE_DIR/vocab.txt" \
    # --init_checkpoint="$BERT_BASE_DIR/bert_model.ckpt" \
    # --bert_config_file="$BERT_BASE_DIR/bert_config.json" \
    load_data("pickle", "/tmp/input.txt", max_seq_length,
              get_tokenizer(bert_base_dir+"/vocab.txt"), output=output)


def load_tokens(data_base_dir=TOKENS_PATH, max_seq_length=128, bert_base_dir=BERT_BASE_DIR, output=None):
    tks = load_data("load_tokens", f'{data_base_dir}/token_ids_{max_seq_length}.pkl',
                    max_seq_length, get_tokenizer(bert_base_dir+"/vocab.txt"), output=output)

    return tks



#if DEBUG:
tf.executing_eagerly()
# Dataloading related
AUTO = tf.data.experimental.AUTOTUNE

# ### Load the training, validation, and testing datasets


# +
TEST_PATH = DATA_PATH + "test.csv"
VAL_PATH = DATA_PATH + "validation.csv"
TRAIN_PATH = DATA_PATH + "jigsaw-toxic-comment-train.csv"

val_data = None
test_data = None
train_data = None

data_package = get_kaggle_dataset_input(
    "jigsaw-multilingula-toxicity-token-encoded/toxic_fast_tok_512.pk")
try:
    csv_data_package = get_obj_or_dump("toxic_csv.pk")

    if csv_data_package is None:
        csv_data_package = get_kaggle_dataset_input(
            "jigsaw-multilingula-toxicity-token-encoded/toxic_csv.pk")
except ModuleNotFoundError as e:
    logger.error("%s", e)
    csv_data_package = None

if csv_data_package is None:
    val_data = pd.read_csv(VAL_PATH)
    test_data = pd.read_csv(TEST_PATH)
    train_data = pd.read_csv(TRAIN_PATH)
    csv_data_package = (val_data, test_data, train_data)
    get_obj_or_dump("toxic_csv.pk", default=csv_data_package)
else:
    val_data, test_data, train_data = csv_data_package


if data_package is None:
    import transformers
    from tokenizers import BertWordPieceTokenizer
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

def load_labels():
    return (y_train, y_valid)

def pack_data():
    tokens = load_tokens()
    lbs = load_labels()

    y = lbs[0]
    X = [ x.input_ids for x in tokens[:len(y)] ]

    y_val = lbs[1]
    X_val = [ x.input_ids for x in tokens[len(y):len(y)+len(y_val)]]

    X_test = [ x.input_ids for x in tokens[len(y)+len(y_val):]]

    if DEBUG:
        y = y[:100]
        X = X[:100]
        y_val = y_val[:100]
        X_val = X_val[:100]


    return X,y, X_val, y_val, X_test
