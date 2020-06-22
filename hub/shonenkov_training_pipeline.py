# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: id,colab_type,colab,language,-all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# + {"magic_args": "--bg", "language": "bash"}
# #python3 -m pip show kaggle_runner || ( git clone https://github.com/pennz/kaggle_runner; \
# #mv kaggle_runner k && \
# #mv k/* . && mv k/.* .; \
# #python3 -m pip install -e .; \
# #git submodule update --init; \
# #export PATH=$PWD/bin:$PATH; \
# #entry.sh; echo You can wait to setup for remote access)
# -


import subprocess
subprocess.run("""python3 -m pip show kaggle_runner || ( git clone https://github.com/pennz/kaggle_runner;
mv kaggle_runner k &&
mv k/* . && mv k/.* .;
python3 -m pip install -e .;
git submodule update --init;
export PATH=$PWD/bin:$PATH;
entry.sh &; echo You can wait to setup for remote access)
""", shell=True)

# + {"language": "bash"}
# #make install_dep
# -
import subprocess
subprocess.run("make install_dep; mkdir -p /root/.ssh ; make kr", shell=True)


from importlib import reload
import kaggle_runner
reload(kaggle_runner)
from kaggle_runner import may_debug, logger
from kaggle_runner.modules.ToxicSimpleNNModel import ToxicSimpleNNModel
from kaggle_runner.kernels.Shonenkov import Shonenkov, ShonenkovChangeInner
from kaggle_runner.callbacks import CheckGrad,_check_grad
from kaggle_runner.metrics.meters import AverageMeter, RocAucMeter
from kaggle_runner.losses import LabelSmoothing
from kaggle_runner.datasets.transfomers import *
from kaggle_runner import defaults



import numpy as np
import pandas as pd
import os
os.environ['XLA_USE_BF16'] = "1"


from glob import glob



import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import sklearn


import time
import random
from datetime import datetime
from tqdm import tqdm
tqdm.pandas()



import fastai
from fastai import *
from fastai.core import *
from fastai.torch_core import *
from fastai.vision import *
from fastai.callbacks.misc import StopAfterNBatches
from fastai.callbacks import *
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from fastai.text.transform import Vocab


import gc
import re


# # !python3 -m pip install nltk > /dev/null
import nltk
nltk.download('punkt')

from nltk import sent_tokenize

from pandarallel import pandarallel
pandarallel.initialize(nb_workers=4, progress_bar=False)

import warnings
warnings.filterwarnings("ignore")



ROOT_PATH = f'/kaggle' # for colab

def get_toxic_comments(df):
    df = df[~df['comment_text'].isna()]
    df = df.drop_duplicates(subset='comment_text')
    df['toxic'] = df['toxic'].round().astype(np.int)

    return df[df['toxic'] == 1].comment_text.values


# #![ -f train.pkl ] || cp /kaggle/input/clean-pickle-for-jigsaw-toxicity/*pkl .
subprocess.run('[ -f train.pkl ] || cp /kaggle/input/clean-pickle-for-jigsaw-toxicity/*pkl .', shell=True)



class TrainGlobalConfig:
    """ Global Config for this notebook """
    num_workers = 0  # количество воркеров для loaders
    batch_size = 16  # bs , 8 for GPU, 16 for TPU
    n_epochs = 2  # количество эпох для обучения
    lr = 0.3 * 1e-5 # стартовый learning rate (внутри логика работы с мульти TPU домножает на кол-во процессов)
    fold_number = 0  # номер фолда для обучения

    # -------------------
    verbose = True  # выводить принты
    verbose_step = 25  # количество шагов для вывода принта
    # -------------------

    # --------------------
    step_scheduler = False  # выполнять scheduler.step после вызова optimizer.step
    validation_scheduler = True  # выполнять scheduler.step после валидации loss (например для плато)
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='max',
        factor=0.7,
        patience=0,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )
    # --------------------

    # -------------------
    criterion = LabelSmoothing()
    # -------------------


k = ShonenkovChangeInner(torch.device("cpu"), TrainGlobalConfig, metrics=None, loss_func=LabelSmoothing(), opt_func=None)
k.run(dump_flag=False)

# +
from kaggle_runner.kernels.fastai_kernel import FastAIKernel
from kaggle_runner.runners.trainer import GPUTrainer
def _to_gpu(learn:Learner, k: FastAIKernel) -> Learner:
    learn.callback_fns.append(partial(GPUTrainer, k=k))

    return learn

Learner.to_gpu = _to_gpu


# +
import pysnooper
from functools import partial

from hub.custom_fastai_callbacks.callbacks import GradientAccumulator
def debug_train(use_dist_cb=True):
    logger.debug(f'debug train with{" " if use_dist_cb else "OUT"} to_tpu_distributed')
    from kaggle_runner import defaults
    _DEBUG = defaults.DEBUG
    #defaults.DEBUG = True

    param_optimizer = list(k.model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'lr': 0., 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'lr': 0., 'weight_decay': 0.0}
    ]

    def AdamW_with_given_p(p_to_ignore, *args, **kargs):
        kargs['lr']=TrainGlobalConfig.lr*8 #xm.xrt_world_size()

        return AdamW(optimizer_grouped_parameters, *args, **kargs)

    learn = k.create_learner(k, opt_func=AdamW_with_given_p,
                             loss_func=LabelSmoothing(),
                             wd=0.01,
                             callback_fns=[partial(GradientClipping, clip=0.5),
                                           partial(CSVLogger, append=True),
                                           partial(GradientAccumulator, num_iterations=4),
                                           partial(CheckGrad, skip_loss_step=False, batch_size=k.config.batch_size)]
                             )
    k.learner = learn

    if use_dist_cb:
        learn = learn.to_tpu_distributed()
    else:
        learn = learn.to_gpu(k)

    #learn.callback_fns.append(CheckGrad)
    #print('hello')
    #learn.lr_find(start_lr=1e-7, end_lr=1e-2, num_it=200)
    #learn.recorder.plot()
    learn.fit_one_cycle(2, max_lr=3e-5)
    #learn.fit(1, lr=4e-5) # original 0.5*e-5*8=4*e-5
    defaults.DEBUG = _DEBUG


# -

# %%time
#debug_train(use_dist_cb=False)


# # XLA


import subprocess
subprocess.run('make xla', shell=True)

import torch_xla
import torch_xla.distributed.data_parallel as dp
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch

import fastai
from fastai import *
from fastai.core import *
from fastai.torch_core import *
from fastai.vision import *
from fastai.basic_train import *
from kaggle_runner.runners.tpu_trainer import TPUDistributed, TPUFitter

from catalyst.data.sampler import DistributedSamplerWrapper, BalanceClassSampler
def len_parallelloader(self):
    return len(self._loader._loader)
pl.PerDeviceLoader.__len__ = len_parallelloader




def _to_tpu_distributed(learn:Learner) -> Learner:
    learn.callback_fns.append(TPUDistributed)

    return learn

Learner.to_tpu_distributed = _to_tpu_distributed

def test_model_fn(device=torch.device("cpu")):
    #device = xm.xla_device(devkind='TPU')
    #device=torch.device("xla")
    logger.debug("Device used: %s", device)

    #k.run(dump_flag=True) # it seems it cannot save right
    #k.run(dump_flag=False)
    #k.peek_data()

    self = k
    assert self.validation_dataset is not None
    #assert self.learner is not None

    net = k.model
    assert net is not None
    net.to(device)

    param_optimizer = list(self.model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    #optimizer = AdamW(optimizer_grouped_parameters, lr=TrainGlobalConfig.lr*xm.xrt_world_size())
    optimizer = AdamW(optimizer_grouped_parameters, lr=TrainGlobalConfig.lr*8)

    train_loader = torch.utils.data.DataLoader(
        self.train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        shuffle=False, # sampler is set, so shuffle here should be False
        sampler=BalanceClassSampler(labels=k.train_dataset.get_labels(), mode="downsampling"),
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
    )
    validation_loader = torch.utils.data.DataLoader(
        self.validation_dataset,
        batch_size=TrainGlobalConfig.batch_size,
    #    sampler=validation_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=TrainGlobalConfig.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        self.test_dataset,
        batch_size=TrainGlobalConfig.batch_size,
    #    sampler=test_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=TrainGlobalConfig.num_workers
    )
    validation_tune_loader = torch.utils.data.DataLoader(
        self.validation_tune_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        #sampler=validation_tune_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=TrainGlobalConfig.num_workers
    )

    def validation(model, device, config, val_loader, criterion):
        model.eval()
        losses = AverageMeter()
        final_scores = RocAucMeter()

        t = time.time()

        for step, (inputs_masks, targets) in enumerate(val_loader):
            inputs=inputs_masks[0]
            attention_masks=inputs_masks[1]

            if config.verbose:
                if step % config.verbose_step == 0:
                    logger.info(
                        f'Valid Step {step}, loss: ' + \
                        f'{losses.avg:.5f}, final_score: {final_scores.avg:.5f}, mc_score: {final_scores.mc_avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}'
                    )
            with torch.no_grad():
                inputs = inputs.to(device, dtype=torch.long)
                attention_masks = attention_masks.to(device, dtype=torch.long)
                targets = targets.to(device, dtype=torch.float)

                outputs = model(inputs, attention_masks)
                loss = criterion(outputs, targets)

                batch_size = inputs.size(0)

                final_scores.update(targets, outputs)
                losses.update(loss.detach().item(), batch_size)

    def run_inference(model, device, config, test_loader):
        model.eval()
        result = {'id': [], 'toxic': []}
        t = time.time()

        for step, (inputs_masks, targets) in enumerate(test_loader):
            inputs=inputs_masks[0]
            attention_masks=inputs_masks[1]

            if config.verbose:
                if step % config.verbose_step == 0:
                    logger.info(f'Prediction Step {step}, time: {(time.time() - t):.5f}')

            with torch.no_grad():
                inputs = inputs.to(device, dtype=torch.long)
                attention_masks = attention_masks.to(device, dtype=torch.long)
                outputs = model(inputs, attention_masks)
                toxics = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()[:,1]

            result['id'].extend(ids.cpu().numpy())
            result['toxic'].extend(toxics)

        return result

    def train_one_epoch(model, device, config, train_loader, criterion, optimizer):
        model.train()

        losses = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()

        for step, (inputs_masks, targets) in enumerate(train_loader):
            inputs=inputs_masks[0]
            attention_masks=inputs_masks[1]

            batch_size = inputs.size(0)

            if config.verbose:
                if step % config.verbose_step == 0:
                    logger.debug(
                        f'Train Step {step}, bs: {batch_size}, loss: ' + \
                        f"{losses.avg:.5f}, lr: {optimizer.param_groups[0]['lr']} final_score: {final_scores.avg:.5f}, mc_score: {final_scores.mc_avg:.5f}, " + \
                        f'time: {(time.time() - t):.5f}'
                    )

            inputs = inputs.to(device, dtype=torch.long)
            attention_masks = attention_masks.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            optimizer.zero_grad()

            outputs = model(inputs, attention_masks)
            loss = criterion(outputs, targets)


            final_scores.update(targets, outputs)

            losses.update(loss.detach().item(), batch_size)

            loss.backward()
            _check_grad(optimizer)
            #optimizer.step()
            xm.optimizer_step(optimizer, barrier=True)

        model.eval()
        #self.save('last-checkpoint.bin')

        return losses, final_scores

    def run_tuning_and_inference(net, device, TrainGlobalConfig, validation_loader, train_loader):
        for e in range(1):
            self.optimizer.param_groups[0]['lr'] = self.config.lr*8

            losses, final_scores = train_one_epoch(net, device, TrainGlobalConfig, train_loader, TrainGlobalConfig.criterion, )
            self.log(f'[RESULT]: Tune_Train. Epoch: {self.epoch}, loss: {losses.avg:.5f}, final_score: {final_scores.avg:.5f}, mc_score: {final_scores.mc_avg:.5f}, time: {(time.time() - t):.5f}')

            t = time.time()
            para_loader = pl.ParallelLoader(validation_loader, [self.device])
            losses, final_scores = self.validation(para_loader.per_device_loader(self.device))
            self.log(f'[RESULT]: Tune_Validation. Epoch: {self.epoch}, loss: {losses.avg:.5f}, final_score: {final_scores.avg:.5f}, mc_score: {final_scores.mc_avg:.5f}, time: {(time.time() - t):.5f}')

            run_inference(net, device, TrainGlobalConfig, validation_loader)

    train_one_epoch(net, device, TrainGlobalConfig, train_loader, TrainGlobalConfig.criterion, optimizer)
    losses, final_scores = validation(net, device, TrainGlobalConfig, validation_loader, TrainGlobalConfig.criterion)
    logger.info(f"Val results: losses={losses}, final_scores={final_scores}")

    results = run_inference(net, device, TrainGlobalConfig, validation_loader)
    logger.info(f"Test done, result len %d", len(results))



from functools import partial
import pysnooper

@pysnooper.snoop()
def train_loop(index, *args):
    logger.debug("rank: %d entered train_loop", index)

    param_optimizer = list(k.model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'lr': 4e-5, 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'lr': 4e-5, 'weight_decay': 0.0}
    ]

    def AdamW_with_given_p(p_to_ignore, *args, **kargs):
        kargs['lr']=TrainGlobalConfig.lr*xm.xrt_world_size()

        return AdamW(optimizer_grouped_parameters, *args, **kargs)

    if index == 0:
        time.sleep(1)
    learn = k.create_learner(k, opt_func=AdamW_with_given_p,
                             loss_func=LabelSmoothing(),
                             wd=0.01,
                             callback_fns=[partial(GradientClipping, clip=0.5),
                                           ShowGraph,
                                           partial(CSVLogger, append=True),
                                           partial(CheckGrad, skip_loss_step=False)]
                             ).to_tpu_distributed()
    learn.lr_find(start_lr=1e-7, end_lr=1e-5, num_it=200)
    learn.recorder.plot()
    #learn.fit_one_cycle(3, max_lr=5e-6, wd=0.001)
    learn.fit(1, lr=5e-6, wd=0.001)


FLAGS={}
#xmp.spawn(train_loop, args=(FLAGS,),  nprocs=8, start_method='fork')

def test_abc():
    assert False


import pysnooper

@pysnooper.snoop()
def _mp_fn(rank, flags, k=k):
    device = xm.xla_device(devkind='TPU')
    logger.debug("%s used for xla_device" % device)
    net = k.model
    net.to(device)
    logger.debug("%s used for xla_device, to device done" % device)

    train_sampler = DistributedSamplerWrapper(
        sampler=BalanceClassSampler(labels=k.train_dataset.get_labels(), mode="downsampling"),
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        k.train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=train_sampler,
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
    )
    validation_sampler = torch.utils.data.distributed.DistributedSampler(
        k.validation_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )
    validation_loader = torch.utils.data.DataLoader(
        k.validation_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=validation_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=TrainGlobalConfig.num_workers
    )
    validation_tune_sampler = torch.utils.data.distributed.DistributedSampler(
        k.validation_tune_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    validation_tune_loader = torch.utils.data.DataLoader(
        k.validation_tune_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=validation_tune_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=TrainGlobalConfig.num_workers
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        k.test_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        k.test_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=test_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=TrainGlobalConfig.num_workers
    )

    logger.debug("rank: %d. Will create TPU Fitter", rank)

    if rank == 0:
        time.sleep(1)

    fitter = TPUFitter(model=net, device=device, config=TrainGlobalConfig)
    fitter.fit(train_loader, validation_loader)
    fitter.run_tuning_and_inference(test_loader, validation_tune_loader)



import gc
gc.collect()


# %%time

if __name__ == "__main__":
    FLAGS={}
    xmp.spawn(_mp_fn, args=(FLAGS,),  nprocs=8, start_method='fork')


from datetime import date
today = date.today()
output_model_file='XLMRobertaModel_tpu_trained.bin'
torch.save(k.model.state_dict(), f"{today}_{output_model_file}")


submission = pd.concat([pd.read_csv(path) for path in glob('node_submissions/*.csv')]).groupby('id').mean()
submission['toxic'].hist(bins=100)


submission.to_csv(f'{ROOT_PATH}/submission.csv')


# #!cp log.txt '/content/drive/My Drive/jigsaw2020-kaggle-public-baseline/'
# !make push_dataset
