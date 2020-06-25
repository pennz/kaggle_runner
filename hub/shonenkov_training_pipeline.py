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

# + {"id": "-5g67DNAUQyh", "colab_type": "code", "colab": {}}
# %load_ext autoreload
# %autoreload 2

# + {"id": "n-Gf0MyQL8Ys", "colab_type": "code", "colab": {}}
#@title Kaggle thing

text = "{   \"username\": \"k1gaggle\",   \"key\": \"721ad312727847f609212568cf015532\",   \"competition\": \"siim-acr-pneumothorax-segmentation\" }" #@param {type:"string"}
#dropdown = '1st option' #@param ["1st option", "2nd option", "3rd option"]
#text_and_dropdown = 'value' #@param ["1st option", "2nd option", "3rd option"] {allow-input: true}
# !mkdir -p ~/.kaggle/
with open('/root/.kaggle/kaggle.json', 'w') as f:
    f.write(text)
#print(dropdown)
#print(text_and_dropdown)

# + {"id": "A6Yz0kroUQy0", "colab_type": "code", "colab": {"base_uri": "https://localhost:8080/", "height": 36}, "magic_args": "--bg", "language": "bash"}
# #python3 -m pip show kaggle_runner || ( git clone https://github.com/pennz/kaggle_runner; \
# #mv kaggle_runner k && \
# #mv k/* . && mv k/.* .; \
# #python3 -m pip install -e .; \
# #git submodule update --init; \
# #export PATH=$PWD/bin:$PATH; \
# #entry.sh; echo You can wait to setup for remote access)


# + {"id": "udc5hHgUUQzD", "colab_type": "code", "colab": {"base_uri": "https://localhost:8080/", "height": 56}}
import subprocess
subprocess.run("""python3 -m pip show kaggle_runner || ( git clone https://github.com/pennz/kaggle_runner;
mv kaggle_runner k && mv k/* . && mv k/.* .;
python3 -m pip install -e .;
git submodule update --init;
export PATH=$PWD/bin:$PATH; entry.sh &
echo You can wait to setup for remote access)
""", shell=True)

import subprocess
subprocess.run("make install_dep; mkdir -p /root/.ssh ; make kr; wait; make xla &", shell=True)


# + {"id": "6lwyNn2ONmLR", "colab_type": "code", "colab": {}}
# !make dd
# !make vim &

# + {"id": "MYBRID_uUQzb", "colab_type": "code", "colab": {}}
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

# + {"id": "e2OOEAbpUQzh", "colab_type": "code", "colab": {}}



# + {"id": "bQfQoj4PUQzl", "colab_type": "code", "colab": {}}
import numpy as np
import pandas as pd
import os
os.environ['XLA_USE_BF16'] = "1"


# + {"id": "jryGiXHWUQzp", "colab_type": "code", "colab": {}}
from glob import glob

# + {"id": "0qyEzTQDUQzu", "colab_type": "code", "colab": {}}



# + {"id": "njd0Gd2TUQz0", "colab_type": "code", "colab": {}}
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import sklearn


# + {"id": "0k4-gbQhUQz5", "colab_type": "code", "colab": {}}
import time
import random
from datetime import datetime
from tqdm import tqdm
tqdm.pandas()

# + {"id": "DFCEM4cYUQz-", "colab_type": "code", "colab": {}}



# + {"id": "63pHGleDUQ0D", "colab_type": "code", "colab": {}}
import fastai
from fastai import *
from fastai.core import *
from fastai.torch_core import *
from fastai.vision import *
from fastai.callbacks.misc import StopAfterNBatches
from fastai.callbacks import *
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from fastai.text.transform import Vocab


# + {"id": "MMbcpmJHUQ0H", "colab_type": "code", "colab": {}}
import gc
import re


# + {"id": "-NTVjOreUQ0L", "colab_type": "code", "colab": {"base_uri": "https://localhost:8080/", "height": 73}}
# # !python3 -m pip install nltk > /dev/null
import nltk
nltk.download('punkt')

# + {"id": "EfgDxZSiUQ0O", "colab_type": "code", "colab": {}}
from nltk import sent_tokenize

# + {"id": "JpTc5nfQUQ0T", "colab_type": "code", "colab": {"base_uri": "https://localhost:8080/", "height": 54}}
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=4, progress_bar=False)

# + {"id": "itAbsNWXUQ0W", "colab_type": "code", "colab": {}}
import warnings
warnings.filterwarnings("ignore")

# + {"id": "c1fHvn3cUQ0a", "colab_type": "code", "colab": {}}



# + {"id": "JHro74jvUQ0d", "colab_type": "code", "colab": {}}
ROOT_PATH = f'/kaggle' # for colab

# + {"id": "IJD6tJMaUQ0i", "colab_type": "code", "colab": {}}
def get_toxic_comments(df):
    df = df[~df['comment_text'].isna()]
    df = df.drop_duplicates(subset='comment_text')
    df['toxic'] = df['toxic'].round().astype(np.int)

    return df[df['toxic'] == 1].comment_text.values


# + {"id": "MpvJJZsbUQ0n", "colab_type": "code", "colab": {"base_uri": "https://localhost:8080/", "height": 36}}
# #![ -f train.pkl ] || cp /kaggle/input/clean-pickle-for-jigsaw-toxicity/*pkl .
subprocess.run('[ -f train.pkl ] || cp /kaggle/input/clean-pickle-for-jigsaw-toxicity/*pkl .', shell=True)

# + {"id": "TfEwN7mhUQ0p", "colab_type": "code", "colab": {}}



# + {"id": "cmkH1Q0qUQ0t", "colab_type": "code", "colab": {}}
class TrainGlobalConfig:
    """ Global Config for this notebook """
    num_workers = 0  # количество воркеров для loaders
    batch_size = 16  # bs , 8 for GPU, 16 for TPU
    n_epochs = 2  # количество эпох для обучения
    lr = 0.5 * 1e-5 # стартовый learning rate (внутри логика работы с мульти TPU домножает на кол-во процессов)
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

# + {"id": "pB36YxahXT6Q", "colab_type": "code", "colab": {}}
from kaggle_runner.kernels.Shonenkov import Shonenkov

# + {"id": "furzEDtzUQ0x", "colab_type": "code", "colab": {"base_uri": "https://localhost:8080/", "height": 184}}
k = Shonenkov(torch.device("cpu"), TrainGlobalConfig, metrics=None, loss_func=LabelSmoothing(), opt_func=None)
k.run(dump_flag=False)

# + {"id": "um8I7KBrY0Ur", "colab_type": "code", "colab": {"base_uri": "https://localhost:8080/", "height": 36}}
# !mkdir ./models_xlmrobert/

# + {"id": "gDt1SW3AUQ02", "colab_type": "code", "colab": {}}
def save_model(self, output_dir="./models_xlmrobert/"):
    model = self.model

    from transformers import WEIGHTS_NAME, CONFIG_NAME
# Step 1: Save a model, configuration and vocabulary that you have fine-tuned

# If we have a distributed model, save only the encapsulated model
# (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
    model_to_save = model.module if hasattr(model, 'module') else model

# If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.backbone.config.to_json_file(output_config_file)
    #tokenizer.save_pretrained(output_dir)

# + {"id": "PN07kHKOUQ06", "colab_type": "code", "colab": {}}
save_model(k)

# + {"id": "0qUwvlFnUQ0-", "colab_type": "code", "colab": {}}
def test_load():
    output_model_file='/kaggle/input/bert-for-toxic-classfication-trained/2020-06-21_XLMRobertaModel_tpu_trained.bin'
    state_dict = torch.load(output_model_file)
    k.model.load_state_dict(state_dict)

    print(k.model)

# + {"id": "9dqdFgwAZQgw", "colab_type": "code", "colab": {"base_uri": "https://localhost:8080/", "height": 1000}}
test_load()
test_predict()

# + {"id": "VXNH-qcCUQ1B", "colab_type": "code", "colab": {}}
from kaggle_runner.kernels.fastai_kernel import FastAIKernel
from kaggle_runner.runners.trainer import GPUTrainer
def _to_gpu(learn:Learner, k: FastAIKernel) -> Learner:
    learn.callback_fns.append(partial(GPUTrainer, k=k))

    return learn

Learner.to_gpu = _to_gpu


# + {"id": "2Z1O-ZLfUQ1D", "colab_type": "code", "colab": {}}
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


# + {"id": "uUzCsw_wUQ1F", "colab_type": "code", "colab": {"base_uri": "https://localhost:8080/", "height": 54}}
# %%time
#debug_train(use_dist_cb=False)


# + {"id": "rO2ay06LUQ1I", "colab_type": "text", "cell_type": "markdown"}
# # XLA


# + {"id": "eeL1PlTnUQ1J", "colab_type": "code", "colab": {"base_uri": "https://localhost:8080/", "height": 36}}
import subprocess
subprocess.run('make xla', shell=True)

# + {"id": "5fUVZUIYUQ1M", "colab_type": "code", "colab": {}}
import torch_xla
import torch_xla.distributed.data_parallel as dp
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch

# + {"id": "qWaNMw-rUQ1O", "colab_type": "code", "colab": {}}
import fastai
from fastai import *
from fastai.core import *
from fastai.torch_core import *
from fastai.vision import *
from fastai.basic_train import *
from kaggle_runner.runners.tpu_trainer import TPUDistributed, TPUFitter

# + {"id": "VomH5RztUQ1T", "colab_type": "code", "colab": {}}
from catalyst.data.sampler import DistributedSamplerWrapper, BalanceClassSampler
def len_parallelloader(self):
    return len(self._loader._loader)
pl.PerDeviceLoader.__len__ = len_parallelloader


# + {"id": "0d8Bzl8DUQ1X", "colab_type": "code", "colab": {}}
def _to_tpu_distributed(learn:Learner) -> Learner:
    learn.callback_fns.append(TPUDistributed)

    return learn

# + {"id": "a6_DgUIyUQ1Z", "colab_type": "code", "colab": {}}
Learner.to_tpu_distributed = _to_tpu_distributed

# + {"id": "wJhxTHZGUQ1c", "colab_type": "code", "colab": {}}
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

        for step, (inputs_masks, ids) in enumerate(test_loader):
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

    #train_one_epoch(net, device, TrainGlobalConfig, train_loader, TrainGlobalConfig.criterion, optimizer)
    #losses, final_scores = validation(net, device, TrainGlobalConfig, validation_loader, TrainGlobalConfig.criterion)
    #logger.info(f"Val results: losses={losses}, final_scores={final_scores}")

    results = run_inference(net, device, TrainGlobalConfig, validation_loader)
    logger.info(f"Test done, result len %d", len(results))


# + {"id": "B_tochnQUQ1e", "colab_type": "code", "colab": {"base_uri": "https://localhost:8080/", "height": 372}}
test_model_fn()


# + {"id": "M_tjbI16UQ1g", "colab_type": "code", "colab": {}}
from functools import partial
import pysnooper

# + {"id": "0nejQ0FwUQ1j", "colab_type": "code", "colab": {}}
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


# + {"id": "JTqfMd9JUQ1k", "colab_type": "code", "colab": {}}
FLAGS={}
#xmp.spawn(train_loop, args=(FLAGS,),  nprocs=8, start_method='fork')


# + {"id": "oh5655pAUQ10", "colab_type": "code", "colab": {}}
import pysnooper

# + {"id": "2YGBLtQ1UQ12", "colab_type": "code", "colab": {}}
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

# + {"id": "8vdV8ftfUQ16", "colab_type": "code", "colab": {}}



# + {"id": "cVRPNFYuUQ18", "colab_type": "code", "colab": {}}
import gc
gc.collect()


# + {"id": "8qPXm5m9UQ1_", "colab_type": "code", "colab": {}}
# %%time

# + {"id": "6x45fTI1UQ2B", "colab_type": "code", "colab": {}}

if __name__ == "__main__":
    FLAGS={}
    xmp.spawn(_mp_fn, args=(FLAGS,),  nprocs=8, start_method='fork')

# + {"id": "JJPxfj1OUQ2E", "colab_type": "code", "colab": {}}
from kaggle_runner.kernels.kernels import KaggleKernelOnlyPredict

# + {"id": "NSC6BrbwUQ2G", "colab_type": "code", "colab": {}}
def only_predict():
    pass


# + {"id": "a_etfRgnUQ2I", "colab_type": "code", "colab": {}}
from datetime import date
today = date.today()
output_model_file='XLMRobertaModel_tpu_trained.bin'
torch.save(k.model.state_dict(), f"{today}_{output_model_file}")


# + {"id": "6Bu-tn3lUQ2J", "colab_type": "code", "colab": {}}
submission = pd.concat([pd.read_csv(path) for path in glob('node_submissions/*.csv')]).groupby('id').mean()
submission['toxic'].hist(bins=100)


# + {"id": "1G896nEvUQ2L", "colab_type": "code", "colab": {}}
submission.to_csv(f'{ROOT_PATH}/submission.csv')


# + {"id": "8RHAQ4PNUQ2N", "colab_type": "code", "colab": {}}
# #!cp log.txt '/content/drive/My Drive/jigsaw2020-kaggle-public-baseline/'
# !make push_dataset
