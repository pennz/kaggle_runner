from glob import glob
import time
import os
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from fastai import *
from fastai.core import *
from fastai.torch_core import *
from fastai.basic_data import *
from fastai.basic_train import LearnerCallback, Learner
from catalyst.data.sampler import DistributedSamplerWrapper, BalanceClassSampler
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.distributed.data_parallel as dp

from kaggle_runner import logger
from kaggle_runner.kernels.fastai_kernel import FastAIKernel
from kaggle_runner.metrics.meters import AverageMeter, RocAucMeter
from kaggle_runner.data_providers import provider
from kaggle_runner.logs import metric_get_log
from kaggle_runner.losses import MixedLoss
from kaggle_runner.metrics.meters import Meter
from kaggle_runner.optimizers import RAdam


class TPUDistributed(LearnerCallback):
    def __init__(self, learn: Learner, debug=True):
        super().__init__(learn)

        self.debug = debug

        if debug:
            self.device = xm.xla_device(devkind='TPU')
            logger.debug("TPUDistributed in DEBUG mode")
            #self.device = xm.xla_device(devkind='CPU')
        else:
            self.device = xm.xla_device(devkind='TPU')
        logger.debug("%s used for xla_device for TPUDistributed" % self.device)

    def on_train_begin(self, **kwargs: Any) -> None:
        self.learn.model = self.learn.model.to(self.device)

        pg = self.learn.opt.opt.param_groups
        pg0pl = pg[0]['params']  # pg0pl[0] is a Parameter
        pg1pl = pg[1]['params']  # pg0pl[0] is a Parameter

        #logger.debug("grad info: %s", raw_opt)
        logger.debug(f"on_train_begin pg0 lr: {pg[0]['lr']}")
        logger.debug(f"on_train_begin pg1 lr: {pg[1]['lr']}")

        if self.debug:
            self.learn.opt.lr = self.learn.opt.lr*xm.xrt_world_size()
            #pg[0]['lr'] *= xm.xrt_world_size() # will do it twice...
            #pg[1]['lr'] *= xm.xrt_world_size()
            logger.debug("opt info: %s\n type: %s",
                         self.learn.opt, type(self.learn.opt))
        else:
            self.learn.opt.lr = self.learn.opt.lr*xm.xrt_world_size()

        logger.debug("%s used for xla_device, to device done" % self.device)

        shuffle = self.data.train_dl.init_kwargs['shuffle'] if hasattr(
            self.data.train_dl, 'init_kwargs') else True
        self.old_sampler_train_dl, self.data.train_dl, self.train_sampler = _change_dl(
            self.k,
            self.data.train_dl, shuffle)

        if hasattr(self.data, 'valid_dl') and self.data.valid_dl is not None:
            self.old_sampler_valid_dl, self.data.valid_dl, self.valid_sampler = _change_dl_val(
                self.k, self.data.valid_dl, shuffle)

    def on_epoch_begin(self, **kwargs: Any) -> None:
        logger.debug("Epoch begins on device %s" % self.device)

        self.old_train_dl = self.data.train_dl
        self.learn.data.train_dl = pl.ParallelLoader(

            self.old_train_dl, [self.device]).per_device_loader(self.device)
        self.learn.data.train_dl.dataset = None  # self.old_train_dl.dataset

        if hasattr(self.data, 'valid_dl') and self.data.valid_dl is not None:
            self.old_valid_dl = self.learn.data.valid_dl
            self.learn.data.valid_dl = pl.ParallelLoader(
                self.old_valid_dl, [self.device]).per_device_loader(self.device)

            self.learn.data.valid_dl.dataset = self.old_valid_dl.dataset
            self.learn.data.valid_dl.dl = self.learn.data.valid_dl._loader._loader

    def on_backward_end(self, **kwargs: Any) -> None:
        # copied from https://github.com/tmabraham/fastai_tpu/blob/8b73018cf705da1a73d9be1f105a8e886051a90c/fastai_v1/tpu_distributed_fastai.py, and needed a fix
        xm.optimizer_step(self.learn.opt.opt, barrier=True)
        #may_debug(True)

        return {'skip_step': True}

    def on_epoch_end(self, **kwargs: Any) -> None:
        self.learn.data.train_dl = self.old_train_dl
        self.learn.data.valid_dl = self.old_valid_dl

    def on_train_end(self, **kwargs: Any) -> None:
        self.learn.data.train_dl = self.old_sampler_train_dl
        self.learn.data.valid_dl = self.old_sampler_valid_dl


class TPUFitter:
    def __init__(self, model, device, config):

        if not os.path.exists('node_submissions'):
            os.makedirs('node_submissions')

        self.config = config
        self.epoch = 0
        self.log_path = 'log.txt'

        self.model = model
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=config.lr*xm.xrt_world_size())
        self.scheduler = config.SchedulerClass(
            self.optimizer, **config.scheduler_params)

        self.criterion = config.criterion
        xm.master_print(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            para_loader = pl.ParallelLoader(train_loader, [self.device])
            losses, final_scores = self.train_one_epoch(
                para_loader.per_device_loader(self.device))

            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, loss: {losses.avg:.5f}, final_score: {final_scores.avg:.5f}, mc_score: {final_scores.mc_avg:.5f}, time: {(time.time() - t):.5f}')

            t = time.time()
            para_loader = pl.ParallelLoader(validation_loader, [self.device])
            losses, final_scores = self.validation(
                para_loader.per_device_loader(self.device))

            self.log(
                f'[RESULT]: Validation. Epoch: {self.epoch}, loss: {losses.avg:.5f}, final_score: {final_scores.avg:.5f}, mc_score: {final_scores.mc_avg:.5f}, time: {(time.time() - t):.5f}')

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=final_scores.mc_avg)

            self.epoch += 1

    def run_tuning_and_inference(self, test_loader, validation_tune_loader):
        for e in range(2):
            self.optimizer.param_groups[0]['lr'] = self.config.lr * \
                xm.xrt_world_size()
            para_loader = pl.ParallelLoader(
                validation_tune_loader, [self.device])
            losses, final_scores = self.train_one_epoch(
                para_loader.per_device_loader(self.device))
            para_loader = pl.ParallelLoader(test_loader, [self.device])
            self.run_inference(para_loader.per_device_loader(self.device))

    def validation(self, val_loader):
        self.model.eval()
        losses = AverageMeter()
        final_scores = RocAucMeter()

        t = time.time()

        for step, (inputs_masks, targets) in enumerate(val_loader):
            inputs = inputs_masks[0]
            attention_masks = inputs_masks[1]

            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    xm.master_print(
                        f'Valid Step {step}, loss: ' +
                        f'{losses.avg:.5f}, final_score: {final_scores.avg:.5f}, mc_score: {final_scores.mc_avg:.5f}, ' +
                        f'time: {(time.time() - t):.5f}'
                    )
            with torch.no_grad():
                inputs = inputs.to(self.device, dtype=torch.long)
                attention_masks = attention_masks.to(
                    self.device, dtype=torch.long)
                targets = targets.to(self.device, dtype=torch.float)

                outputs = self.model(inputs, attention_masks)
                loss = self.criterion(outputs, targets)

                batch_size = inputs.size(0)

                final_scores.update(targets, outputs)
                losses.update(loss.detach().item(), batch_size)

        return losses, final_scores

    def train_one_epoch(self, train_loader):
        self.model.train()

        losses = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()

        for step, (inputs_masks, targets) in enumerate(train_loader):
            inputs = inputs_masks[0]
            attention_masks = inputs_masks[1]
            batch_size = inputs.size(0)

            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    self.log(
                        f'Train Step {step}, loss: ' +
                        f"{losses.avg:.5f}, final_score: {final_scores.avg:.5f}, mc_score: {final_scores.mc_avg:.5f}, " +
                        f'time: {(time.time() - t):.5f}'
                    )

            inputs = inputs.to(self.device, dtype=torch.long)
            attention_masks = attention_masks.to(self.device, dtype=torch.long)
            targets = targets.to(self.device, dtype=torch.float)

            self.optimizer.zero_grad()

            outputs = self.model(inputs, attention_masks)
            loss = self.criterion(outputs, targets)

            final_scores.update(targets, outputs)

            losses.update(loss.detach().item(), batch_size)

            loss.backward()
            #logger.info("step: %d, loss: %f", step, loss)

            xm.optimizer_step(self.optimizer)

            if self.config.step_scheduler:
                self.scheduler.step()

        self.model.eval()
        self.save('last-checkpoint.bin')

        return losses, final_scores

    def run_inference(self, test_loader):
        self.model.eval()
        result = {'id': [], 'toxic': []}
        t = time.time()

        for step, (inputs_masks, ids) in enumerate(test_loader):
            inputs = inputs_masks[0]
            attention_masks = inputs_masks[1]

            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    xm.master_print(
                        f'Prediction Step {step}, time: {(time.time() - t):.5f}')

            with torch.no_grad():
                inputs = inputs.to(self.device, dtype=torch.long)
                attention_masks = attention_masks.to(
                    self.device, dtype=torch.long)
                outputs = self.model(inputs, attention_masks)
                toxics = nn.functional.softmax(
                    outputs, dim=1).data.cpu().numpy()[:, 1]

            result['id'].extend(ids.cpu().numpy())
            result['toxic'].extend(toxics)

        result = pd.DataFrame(result)
        node_count = len(glob('node_submissions/*.csv'))
        result.to_csv(
            f'node_submissions/submission_{node_count}_{datetime.utcnow().microsecond}_{random.random()}.csv', index=False)

    def save(self, path):
        xm.save(self.model.state_dict(), path)

    def log(self, message):
        if self.config.verbose:
            xm.master_print(message)
        with open(self.log_path, 'a+') as logger:
            xm.master_print(f'{message}', logger)
