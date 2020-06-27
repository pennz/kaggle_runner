import os
import pickle
import re

import numpy as np
import pandas as pd
import tensorflow as tf

from kaggle_runner import may_debug, logger
from kaggle_runner.defaults import DEBUG, LOAD_BERT_DATA
from kaggle_runner.utils.kernel_utils import (get_kaggle_dataset_input,
                                              get_obj_or_dump)
#from kaggle_runner.utils.tpu import (strategy, tpu_resolver)
from hub.bert.extract_features import load_data, get_tokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from fastai import *
from fastai.core import *


TOKENS_PATH = "/kaggle/input/jigsaw-toxic-token-ids-for-bert"
PRETRAIND_PICKLE_AND_MORE='/kaggle/input/toxic-multilang-trained-torch-model'
TRAIN_LEN = 0

strategy=None
tpu_resolver=None
BATCH_SIZE = 32 * 2
MAX_LENGTH = 224
X,y, x_valid, y_valid, X_test = None,None,None,None,None

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

# +
train_dataset = None
test_dataset = None
valid_dataset = None

if LOAD_BERT_DATA:
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

def xlmr_data():
    xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')
    xlmr.eval()

    xlmr.encode('Hello world!')
    del xlmr

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

def onehot(size, target, aux=None):
    if aux is not None:
        vec = np.zeros(size+len(aux), dtype=np.float32)
        vec[target] = 1.
        vec[2:] = aux
        vec = torch.tensor(vec, dtype=torch.float32)
    else:
        vec = torch.zeros(size, dtype=torch.float32)
        vec[target] = 1.

    return vec

class DatasetRetriever(Dataset):
    def __init__(self, labels_or_ids, comment_texts, langs,
                 severe_toxic=None, obscene=None, threat=None, insult=None, identity_hate=None,
                 use_train_transforms=False, test=False, use_aux=True, transformers=None):
        self.test = test
        self.labels_or_ids = labels_or_ids
        self.comment_texts = comment_texts
        self.langs = langs
        self.severe_toxic = severe_toxic
        self.obscene = obscene
        self.threat = threat
        self.insult = insult
        self.identity_hate = identity_hate
        self.use_train_transforms = use_train_transforms
        self.aux = None
        assert transformers is not None
        self.transformers = transformers
        self.vocab = None

        if use_aux:
            self.aux = [self.severe_toxic, self.obscene, self.threat, self.insult, self.identity_hate]

    def get_tokens(self, text):
        encoded = self.transformers['tokenizer'].encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            pad_to_max_length=True
        )

        return encoded['input_ids'], encoded['attention_mask']

    def __len__(self):
        return self.comment_texts.shape[0]

    def __getitem__(self, idx):
        text = self.comment_texts[idx]
        lang = self.langs[idx]

        if self.severe_toxic is None:
            aux = [0., 0., 0., 0., 0.]
        else:
            aux = [self.severe_toxic[idx], self.obscene[idx], self.threat[idx], self.insult[idx], self.identity_hate[idx]]


        label = self.labels_or_ids[idx]

        if self.use_train_transforms and (not self.test):
            text, _ = self.transformers['train_transforms'](data=(text, lang))['data']
            tokens, attention_mask = self.get_tokens(str(text))
            token_length = sum(attention_mask)

            if token_length > 0.8*MAX_LENGTH:
                text, _ = self.transformers['shuffle_transforms'](data=(text, lang))['data']
            elif token_length < 60:
                text, label = self.transformers['synthesic_transforms_often'](data=(text, label))['data']
            else: # will not need to use transforms
                #text, label = synthesic_transforms_low(data=(text, label))['data']
                pass

        # TODO add language detection and shuffle
        # https://pypi.org/project/langdetect/
        # if self.use_train_transforms and self.test:
        #    text, _ = train_transforms(data=(text, lang))['data']
        #    tokens, attention_mask = self.get_tokens(str(text))
        #    token_length = sum(attention_mask)

        #    if token_length > 0.8*MAX_LENGTH:
        #        text, _ = shuffle_transforms(data=(text, lang))['data']
        # to tensors
        tokens, attention_mask = self.get_tokens(str(text))
        tokens, attention_mask = torch.tensor(tokens), torch.tensor(attention_mask)

        if self.test:  # for test, return id TODO TTA
            return [tokens, attention_mask], self.labels_or_ids[idx]

        # label might be changed
        target = onehot(2, label, aux=aux)

        return [tokens, attention_mask], target

    def get_labels(self):
        return list(np.char.add(self.labels_or_ids.astype(str), self.langs))
