from unittest import TestCase

import ripdb
from kaggle_runner.datasets.bert import (BATCH_SIZE, TRAIN_LEN, train_dataset,
                                         val_data, valid_dataset, x_valid,
                                         y_valid)
from kaggle_runner.kernels.bert import (bert_cbs,
                                        build_distilbert_model_singleton)
from kaggle_runner.utils.visualizer import visualize_model_preds


class Test_distilbert_model(TestCase):
    @classmethod
    def setup_class(cls):
        cls.model_distilbert = build_distilbert_model_singleton()

    @classmethod
    def teardown_class(cls):
        del cls.model_distilbert

    def test_summary(self):
        ripdb.set_trace()
        self.model_distilbert.summary()

    def test_fit(self):
        train_history = self.model_distilbert.fit(
            train_dataset,
            steps_per_epoch=TRAIN_LEN/BATCH_SIZE,
            validation_data=valid_dataset,
            callbacks=bert_cbs,
            epochs=8
        )
# ### Visualize model architecture

# SVG(tf.keras.utils.model_to_dot(model_distilbert, drpi=70).create(prog='dot', format='svg'))

    def test_visualize(self):
        # model_distilbert.summary()
        visualize_model_preds(self.model_distilbert, val_data, x_valid, y_valid,
                              indices=[2,3, 5, 6, 7, 8, 1, 4])
