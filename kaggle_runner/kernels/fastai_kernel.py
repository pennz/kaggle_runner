from fastai import *
from fastai import vision
from fastai.basic_data import *
from fastai.basic_train import Learner
from fastai.callbacks import CSVLogger
from fastai.core import *
from fastai.torch_core import *

from . import KaggleKernel
from .. import logger


class FastAIKernel(KaggleKernel):
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
