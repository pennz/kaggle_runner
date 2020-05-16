import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import transformers
from kaggle_runner.callbacks import ReduceLROnPlateauLogCBs
from kaggle_runner.datasets.bert import (train_dataset, val_data,
                                         valid_dataset, x_valid, y_valid)
from kaggle_runner.utils.tpu import strategy
from kaggle_runner.utils.visualizer import visualize_model_preds

# ### DistilBERT
#
# DistilBERT is a lighter version of BERT (a very complex model) which uses
# fewer weights and achieves similar accuracies on several tasks with much lower
# training times. For this notebook, I will be using DistilBERT as it is easier
# to train in less time. The approach can be summarized with the flowchart below:
#
# <center><img src="https://i.imgur.com/6AGu9a4.png" width="315px"></center>
N_STEPS = 100
calls = ReduceLROnPlateauLogCBs((x_valid, y_valid))
EPOCHS = 2

# ### Define the model

def build_distilbert_model(transformer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    cls_token = Dense(500, activation="elu")(cls_token)
    cls_token = Dropout(0.1)(cls_token)
    out = Dense(1, activation='sigmoid')(cls_token)

    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(Adam(lr=1.5e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# ### Build the model and check summary

# +
with strategy.scope():
    transformer_layer = transformers.TFDistilBertModel.\
    from_pretrained('distilbert-base-multilingual-cased')
    model_distilbert = build_distilbert_model(transformer_layer, max_len=512)

model_distilbert.summary()
# -

# ### Train the model

train_history = model_distilbert.fit(
    train_dataset,
    steps_per_epoch=N_STEPS,
    validation_data=valid_dataset,
    callbacks = calls,
    epochs=EPOCHS
)

# ### Visualize model architecture

# SVG(tf.keras.utils.model_to_dot(model_distilbert, drpi=70).create(prog='dot', format='svg'))

# ### Visualize model predictions

# + {"_kg_hide-input": true}
# def visualize_model_preds(model,val_data, x_valid, y_valid, indices=[0, 17, 1, 24]):
visualize_model_preds(model_distilbert, val_data, x_valid, y_valid)
# -
