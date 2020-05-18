import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import transformers
from kaggle_runner.callbacks import ReduceLROnPlateauLogCBs
from kaggle_runner.datasets.bert import x_valid, y_valid
from kaggle_runner.metrics.metrics import matthews_correlation_aux_stripper
from kaggle_runner.utils.tpu import strategy

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

    return _build_distilbert_model(transformer, max_len=512)

def _build_distilbert_model(transformer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]
    cls_token = Dense(500, activation="elu")(cls_token)
    cls_token = Dropout(0.2)(cls_token)
    out = Dense(6, activation='sigmoid')(cls_token)

    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(Adam(lr=1.5e-5),
                  loss='binary_crossentropy',
                  metrics=[matthews_correlation_aux_stripper])

    return model


def build_distilbert_model_singleton():
    global __model_distilbert

    if __model_distilbert is None:
        if strategy is None:
            transformer_layer = transformers.TFDistilBertModel.\
            from_pretrained('distilbert-base-multilingual-cased')
            __model_distilbert = _build_distilbert_model_adv(transformer_layer, max_len=512)
        else:
            with strategy.scope():
                transformer_layer = transformers.TFDistilBertModel.\
                from_pretrained('distilbert-base-multilingual-cased')
                __model_distilbert = _build_distilbert_model_adv(transformer_layer, max_len=512)

    return __model_distilbert
