import subprocess

import ripdb

from kaggle_runner import may_debug
from kaggle_runner.datasets.bert import (BATCH_SIZE, TRAIN_LEN, train_dataset,
                                         val_data, valid_dataset, x_valid,
                                         y_valid)
from kaggle_runner.kernels.bert import (bert_cbs,
                                        build_distilbert_model_singleton)
from kaggle_runner.utils.visualizer import visualize_model_preds


class Test_distilbert_model:
    @classmethod
    def setup_class(cls):
        # subprocess.run("make ripdbrv &", shell=True)
        cls.model_distilbert = None

    @classmethod
    def teardown_class(cls):
        subprocess.run("pkill -f \"make ripdbrv\"", shell=True)
        try:
            del cls.model_distilbert
        except Exception as e:
            print(e)

    def test_data(self):
        assert val_data is not None
        assert train_dataset is not None

    def test_summary(self):
        may_debug()
        self.model_distilbert_dev = build_distilbert_model_singleton()

        self.model_distilbert.summary()
        assert True

    def test_fit_adv(self):
        # self.model_distilbert_dev = build_distilbert_model_singleton(model_type="1st")
        train_history = self.model_distilbert_dev.fit(
            train_dataset,
            steps_per_epoch=TRAIN_LEN/BATCH_SIZE,
            validation_data=valid_dataset,
            callbacks=bert_cbs,
            epochs=8
        )

    def test_fit(self):
        train_history = self.model_distilbert.fit(
            train_dataset,
            steps_per_epoch=TRAIN_LEN/BATCH_SIZE/10, # just try a little
            validation_data=valid_dataset,
            callbacks=bert_cbs,
            epochs=1
        )
# ### Visualize model architecture

# SVG(tf.keras.utils.model_to_dot(model_distilbert, drpi=70).create(prog='dot', format='svg'))

    def test_visualize(self):
        # model_distilbert.summary()
        visualize_model_preds(self.model_distilbert, val_data, x_valid, y_valid,
                              indices=[2,3, 5, 6, 7, 8, 1, 4])
