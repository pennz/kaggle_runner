import subprocess

from kaggle_runner import logger, may_debug
from kaggle_runner.datasets.bert import (BATCH_SIZE, TRAIN_LEN, train_dataset,
                                         val_data, valid_dataset, x_valid,
                                         y_valid)
from kaggle_runner.defaults import DEBUG
from kaggle_runner.kernels.bert import (bert_cbs,
                                        build_distilbert_model_singleton,
                                        get_test_result)
from kaggle_runner.utils.visualizer import visualize_model_preds


class Test_distilbert_model:
    @classmethod
    def setup_class(cls):
        # subprocess.run("make ripdbrv &", shell=True)

        if DEBUG:
            cls.model_distilbert = build_distilbert_model_singleton(140)
        else:
            cls.model_distilbert = build_distilbert_model_singleton(512)

    @classmethod
    def teardown_class(cls):
        #subprocess.run("pkill -f \"make ripdbrv\"", shell=True)
        try:
            del cls.model_distilbert
        except Exception as e:
            print(e)
        logger.debug("tear down test %s", "Test_distilbert_model")

    def setup_method(self, method):
        logger.debug("setup for method %s", method)

    def teardown_method(self, method):
        logger.debug("teardown method %s", method)

    def test_data(self):
        assert val_data is not None
        assert train_dataset is not None

    def test_summary(self):
        self.setup_class()
        self.model_distilbert.summary()
        assert True

    def test_fit_adv(self):
        # self.model_distilbert_dev = build_distilbert_model_singleton(model_type="1st")

        if DEBUG:
            steps = 10
            epochs = 1
        else:
            steps = TRAIN_LEN//BATCH_SIZE
            logger.debug("Train len %s, batch size %s", TRAIN_LEN, BATCH_SIZE)
            epochs = 1
        logger.debug("Every epoch, steps is %s", steps)

        train_history = self.model_distilbert.fit(
            train_dataset,
            steps_per_epoch=steps,
            validation_data=valid_dataset,
            callbacks=bert_cbs,
            epochs=epochs
        )
    def test_result(self):
        get_test_result(self)

    def test_visualize(self):
        # model_distilbert.summary()
        visualize_model_preds(self.model_distilbert, val_data, x_valid, y_valid,
                              indices=[2,3, 5, 6, 7, 8, 1, 4])

if __name__ == "__main__":
    tt = Test_distilbert_model()
    tt.test_summary()
    tt.test_fit_adv()
    tt.test_visualize()
    tt.test_result()
