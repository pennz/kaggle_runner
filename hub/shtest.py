from kaggle_runner.kernels.kernel import KaggleKernelOnlyPredict

class ToxicPredictModel(KaggleKernelOnlyPredict):
    def __init__(self, model_path):
        super(ToxicPredictModel, self).__init__(model_path)
        self.only_predict = True

    def build_and_set_model(self):
        """load pretrained one"""
        pass

    def prepare_train_dev_data(self):
        pass

    #def prepare_test_data(self, data_config=None):
    #    pass

    def check_predict_details(self):
        pass

    def peek_data(self):
        pass

def only_predict():
    pass

def test_init():
    k = ToxicPredictModel(".")
    assert k is not None
