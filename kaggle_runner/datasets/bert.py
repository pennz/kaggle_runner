import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf

import ripdb
import transformers
from kaggle_runner import may_debug
from kaggle_runner.utils.tpu import BATCH_SIZE
from tokenizers import BertWordPieceTokenizer

# Dataloading related
AUTO = tf.data.experimental.AUTOTUNE

# ### Load the training, validation, and testing datasets

DATA_PATH = "/kaggle/input/jigsaw-multilingual-toxic-comment-classification/"
os.listdir(DATA_PATH)

# +
TEST_PATH = DATA_PATH + "test.csv"
VAL_PATH = DATA_PATH + "validation.csv"
TRAIN_PATH = DATA_PATH + "jigsaw-toxic-comment-train.csv"

val_data = pd.read_csv(VAL_PATH)
test_data = pd.read_csv(TEST_PATH)
train_data = pd.read_csv(TRAIN_PATH)
TRAIN_LEN = len(train_data)
# -

tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

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
    text = text.map(lambda x: re.sub('\\n',' ',str(x)))
    text = text.map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
    text = text.map(lambda x: re.sub("\(http://.*?\s\(http://.*\)",'',str(x)))

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

# TODO just save it to disk or dataset for faster startup
y_valid = val.toxic.values
# y_train = train.toxic.values  # TODO add aux data
### Define training, validation, and testing datasets

y_train=np.stack(
[train.toxic.values, train.severe_toxic.values, train.obscene.values,
 train.threat.values, train.insult.values, train.identity_hate.values]
).T
may_debug()
# +
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048*8)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .batch(BATCH_SIZE)
)


# -
