from kaggle_runner.kernels import bert_torch
from kaggle_runner.datasets.bert import pack_data

from kaggle_runner import logger, may_debug

class Test_bert_multi_lang:
    @classmethod
    def setup_class(cls):
        try:
            cls.model = bert_torch.get_trained_model()
        except RuntimeError:
            cls.model = None
        cls.data=pack_data()
        logger.debug("Start Test bert multi lang")

    def setup_method(self, method):
        logger.debug("setup for method %s", method)

    def teardown_method(self, method):
        logger.debug("teardown method %s", method)

    def test_continue_train(self):
        bert_torch.for_pytorch(self.data, phase="continue_train", model=self.model)

    def test_result(self):
        bert_torch.get_test_result(self, self.data[-1])

if __name__ == "__main__":
    Test_bert_multi_lang.setup_class()
    t = Test_bert_multi_lang()
    t.test_continue_train()
    t.test_result()
