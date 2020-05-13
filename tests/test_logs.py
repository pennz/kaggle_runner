from unittest import TestCase

from kaggle_runner.logs import NBatchProgBarLogger


class TestLogs(TestCase):
    @classmethod
    def setup_class(cls):
        "prepare a dummy model for logger to show"
        # TODO set up a dummy model / or just use imagenette
        cls.model = None

    @classmethod
    def teardown_class(cls):
        print("teardown_class called once for the class")

    def test_NBatchProgBarLogger(self):
        assert self.model is not None
        self.BS = 100

        self.model.fit(
            [X_train_aux, X_train_main],
            Y_train,
            batch_size=BS,
            verbose=0,
            callbacks=[NBatchProgBarLogger()],
        )
