from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.callbacks import (Callback, CSVLogger, ModelCheckpoint,
                                        ReduceLROnPlateau)

from kaggle_runner import logger
from kaggle_runner.metrics.meters import AverageMeter,RocAucMeter


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)

            if y_pred.size > self.y_val.size:
                y_pred = y_pred[:,0]
            score = roc_auc_score(self.y_val, y_pred)
            print(
                "\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))
            logger.debug(
                "\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))


def ReduceLROnPlateauLogCBs(validation_data):
    cb = []

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.3, patience=3,
                                       verbose=1, mode='auto',
                                       epsilon=0.0001, cooldown=1, min_lr=0.000001)
    cb.append(reduceLROnPlat)
    log = CSVLogger('log.csv')
    cb.append(log)

    RocAuc = RocAucEvaluation(validation_data=validation_data, interval=1)
    cb.append(RocAuc)
    # ckpt = ModelCheckpoint("toxic.hdf5", save_best_only=True, verbose=1) bert get_config not implemented, cannot save
    # cb.append(ckpt)

    return cb

import torch
from fastai.core import Any
from fastai.basic_train import LearnerCallback, Learner

def _check_grad(raw_opt):
    pg = raw_opt.param_groups
    pg0pl = pg[0]['params'] # pg0pl[0] is a Parameter
    pg1pl = pg[1]['params'] # pg0pl[0] is a Parameter

    with torch.no_grad():
        #norms = torch.tensor([torch.norm(p) for p in pg0pl])
        #may_debug()
        #logger.debug("%s", pg0pl[0].grad)
        #logger.debug("%s", pg0pl[0].data)
        normsg = torch.tensor([torch.norm(p.grad) for p in pg0pl[:10] if p.grad is not None])
        #logger.debug("params info pg0: norm std(%f) mean(%f)", *torch.std_mean(norms))
        logger.debug("grad info pg0: norm std(%f) mean(%f)", *torch.std_mean(normsg))

        #norms1 = torch.tensor([torch.norm(p) for p in pg1pl])
        norms1g = torch.tensor([torch.norm(p.grad) for p in pg1pl[:10] if p.grad is not None])
        #logger.debug("params info pg1: norm std(%f) mean(%f)", *torch.std_mean(norms1))
        logger.debug("grad info pg1: norm std(%f) mean(%f)", *torch.std_mean(norms1g))

class CheckGrad(LearnerCallback):
    def __init__(self, learn:Learner, skip_loss_step=False, batch_size=16):
        super().__init__(learn)
        self.skip_loss_step = skip_loss_step
        logger.debug("Init Callback CheckGrad with skip_loss_step: " +str(self.skip_loss_step))
        self.losses = None
        self.final_scores = None
        self.batch_size = batch_size

    def on_train_begin(self, **kwargs:Any)->None:
        self.losses = AverageMeter()
        self.final_scores = RocAucMeter()

    def on_backward_begin(self, **kwargs:Any)->None:
        #print(kwargs.keys())
        """dict_keys(['epoch', 'iteration', 'num_batch', 'skip_validate',
        'n_epochs', 'pbar', 'metrics', 'stop_training', 'last_input',
        'last_target', 'train', 'stop_epoch', 'skip_step', 'skip_zero',
        'skip_bwd', 'last_output', 'last_loss', 'smooth_loss'])
        """
        pg = self.learn.opt.opt.param_groups
        #logger.debug("grad info: %s", raw_opt)
        logger.debug(f"on_backward_begin lr: {pg[0]['lr']}")
        logger.debug("itr: %d, num_batch: %d, last loss: %f, smooth_loss: %f",
                     kwargs['iteration'], kwargs['num_batch'],
                     kwargs['last_loss'], kwargs['smooth_loss'])

        self.final_scores.update(kwargs['last_target'], kwargs['last_output'])
        self.losses.update(kwargs['last_loss'].detach().item(), self.batch_size)
        logger.debug(f"loss_avg: {self.losses.avg:.5f}, lr_pg0:"
                     f"{pg[0]['lr']}, lr_pg1: {pg[1]['lr']}final_score:"
                     f"{self.final_scores.avg:.5f}, mc_score:"
                     f"{self.final_scores.mc_avg:.5f}")

    def on_backward_end(self, **kwargs:Any)->None:
        raw_opt = self.learn.opt.opt
        _check_grad(raw_opt)

        return {'skip_step': self.skip_loss_step}
