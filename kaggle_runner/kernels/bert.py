import os

import pandas as pd
import tensorflow as tf
import transformers
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from kaggle_runner import may_debug
from kaggle_runner.callbacks import ReduceLROnPlateauLogCBs
from kaggle_runner.datasets.bert import (DATA_PATH, test_dataset, x_valid,
                                         y_valid)
from kaggle_runner.metrics.metrics import matthews_correlation
from kaggle_runner.utils.tpu import strategy
from kaggle_runner.utils.wrapper import size_decorator

# ### learn from 1st place solution
# Custom head for BERT, XLNet and GPT2 and Bucket Sequencing Collator
# Auxiliary tasks for models -> to add
# Custom mimic loss -> done, need test
# SWA and checkpoint ensemble
# Rank average ensemble of 2x XLNet, 2x BERT and GPT2 medium


# ### DistilBERT
#
# DistilBERT is a lighter version of BERT (a very complex model) which uses
# fewer weights and achieves similar accuracies on several tasks with much lower
# training times. For this notebook, I will be using DistilBERT as it is easier
# to train in less time. The approach can be summarized with the flowchart below:
#
# <center><img src="https://i.imgur.com/6AGu9a4.png" width="315px"></center>
bert_cbs = ReduceLROnPlateauLogCBs((x_valid, y_valid))

# ### Define the model
__model_distilbert = None


def _build_distilbert_model_adv(transformer, max_len=512):
    """
    _build_distilbert_model_adv just follow the good result and try to replicate
    https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/103280
    """

    return _build_distilbert_model(transformer, max_len=max_len)

def _build_distilbert_model(transformer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]
    cls_token = Dense(500, activation="elu")(cls_token)
    cls_token = Dropout(0.2)(cls_token)
    out = Dense(6, activation='sigmoid')(cls_token)

    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(Adam(lr=1.5e-5),
                  loss=size_decorator(BinaryCrossentropy(reduction=
                                                         tf.keras.losses.Reduction.NONE
                                                         , label_smoothing=0.2)),
                  #'binary_crossentropy',
                  metrics=[size_decorator(binary_accuracy),size_decorator(matthews_correlation)])

    return model


def build_distilbert_model_singleton(max_len):
    global __model_distilbert

    #if __model_distilbert is None:

    if True:
        if strategy is None:
            transformer_layer = transformers.TFDistilBertModel.\
            from_pretrained('distilbert-base-multilingual-cased')
            __model_distilbert = _build_distilbert_model_adv(transformer_layer, max_len=max_len)
        else:
            with strategy.scope():
                transformer_layer = transformers.TFDistilBertModel.\
                from_pretrained('distilbert-base-multilingual-cased')
                __model_distilbert = _build_distilbert_model_adv(transformer_layer, max_len=max_len)

    return __model_distilbert

def get_test_result(self, test_dataset=test_dataset, data_path=DATA_PATH):
    sub = pd.read_csv(os.path.join(data_path , 'sample_submission.csv'))
    pred = self.model_distilbert.predict(test_dataset, verbose=1)

    if len(pred.shape) <= 1 or pred.shape[1] > 1:
        pred = pred[:,0]
    sub['toxic'][:len(pred)] = pred
    sub.to_csv('submission.csv', index=False)
