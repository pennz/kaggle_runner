from kaggle_runner.kernels import bert_torch
from kaggle_runner.datasets.bert import pack_data

from kaggle_runner import logger, may_debug

class Test_bert_multi_lang:
    @classmethod
    def setup_class(cls):
        cls.model = bert_torch.get_trained_model()
        logger.debug("Start Test bert multi lang")

    def setup_method(self, method):
        logger.debug("setup for method %s", method)

    def teardown_method(self, method):
        logger.debug("teardown method %s", method)

    def test_result(self):
        data=pack_data()
        bert_torch.get_test_result(self, data[-1])
