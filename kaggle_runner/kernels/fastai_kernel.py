from fastai import *
from fastai import vision
from fastai.basic_data import *
from fastai.basic_train import Learner
from fastai.callbacks import CSVLogger
from fastai.core import *
from fastai.torch_core import *

from .kernel import KaggleKernel
from .. import logger


class FastAIKernel(KaggleKernel):
    """Can't instantiate abstract class FastAIKernel with abstract methods
    build_and_set_model, check_predict_details, peek_data, set_loss,
    set_metrics
    """

    def build_and_set_model(self):
        self.model = None

    def check_predict_details(self):
        assert False

    def peek_data(self):
        assert False

    def set_loss(self, loss_func):
        self.model_loss = loss_func

    def set_metrics(self, metrics):
        self.model_metrics = metrics

    def __init__(self, **kargs):
        super(FastAIKernel, self).__init__(logger=logger)
        self.developing = True
        self.learner = None

        for required in ['loss_func', 'metrics']:
            assert required in kargs

        for k,v in kargs.items():
            setattr(self, k, v)

    def setup_learner(self, data, model, opt_func, loss_func, metrics):
        self.data = self.wrap_databunch()

        data = self.data
        model = self.model
        opt_func = self.opt_func if hasattr(self, 'opt_func') and self.opt_func is not None else AdamW
        loss_func = self.model_loss
        metrics = self.model_metrics

        return Learner(data, model, opt_func, loss_func, metrics, bn_wd=False)


def test_learner_init():
    l = FastAIKernel(loss_func=None, metrics=None)
    assert l is not None

def test_learner_fit():
    k = FastAIKernel()
    k.leaner.fit_one_cycle() # just test basic function
