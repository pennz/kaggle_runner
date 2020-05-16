from unittest import TestCase

from kaggle_runner.datasets.bert import (train_dataset, val_data,
                                         valid_dataset, x_valid, y_valid)
from kaggle_runner.kernels.bert import calls, model_distilbert
from kaggle_runner.utils.visualizer import visualize_model_preds


class Test_distilbert_model(TestCase):
    def test_build_distilbert_model(self):
        self.fail()

    def test_summary(self):
        model_distilbert.summary()

    def test_fit(self):
        train_history = model_distilbert.fit(
            train_dataset,
            steps_per_epoch=24,
            validation_data=valid_dataset,
            callbacks = calls,
            epochs=1
        )
# ### Visualize model architecture

# SVG(tf.keras.utils.model_to_dot(model_distilbert, drpi=70).create(prog='dot', format='svg'))

# ### Visualize model predictions

# + {"_kg_hide-input": true}
# def visualize_model_preds(model,val_data, x_valid, y_valid, indices=[0, 17, 1, 24]):
# -
    def test_visualize(self):
        # model_distilbert.summary()
        visualize_model_preds(model_distilbert, val_data, x_valid, y_valid)
