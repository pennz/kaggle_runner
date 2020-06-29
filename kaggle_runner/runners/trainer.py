from glob import glob
import time
import os

import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from fastai import *
from fastai.core import *
from fastai.torch_core import *
from fastai.basic_data import *
from fastai.basic_train import LearnerCallback, Learner
from catalyst.data.sampler import DistributedSamplerWrapper, BalanceClassSampler
from datetime import datetime

from kaggle_runner import logger
from kaggle_runner.kernels.fastai_kernel import FastAIKernel
from kaggle_runner.metrics.meters import AverageMeter, RocAucMeter
from kaggle_runner.data_providers import provider
from kaggle_runner.logs import metric_get_log
from kaggle_runner.losses import MixedLoss
from kaggle_runner.metrics.meters import Meter
from kaggle_runner.optimizers import RAdam


def _change_dl(k, dl, shuffle):
    old_dl = dl
    train_sampler = DistributedSamplerWrapper(
        sampler=BalanceClassSampler(
            labels=k.train_dataset.get_labels(), mode="downsampling"),
        num_replicas=8,  # xm.xrt_world_size(),
        rank=0,  # xm.get_ordinal(), it only get 1/8 data ....
        shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        k.train_dataset,
        batch_size=k.config.batch_size,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
        num_workers=k.config.num_workers,
    )
    new_dl = train_loader

    return old_dl, new_dl, train_sampler


def _change_dl_val(k, dl, shuffle):
    old_dl = dl
    validation_sampler = torch.utils.data.distributed.DistributedSampler(
        k.validation_dataset,
        num_replicas=8,  # xm.xrt_world_size(),
        rank=0,  # xm.get_ordinal(),
        shuffle=False
    )
    validation_loader = torch.utils.data.DataLoader(
        k.validation_dataset,
        batch_size=k.config.batch_size,
        sampler=validation_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=k.config.num_workers
    )

    return old_dl, validation_loader, validation_sampler


class Trainer:
    """This class takes care of training and validation of our model"""

    def __init__(self, model, data_folder, df_path):
        self.fold = 1
        self.total_folds = 5
        self.num_workers = 4
        self.batch_size = {"train": 4, "val": 4}
        self.accumulation_steps = 32 // self.batch_size["train"]
        self.lr = 5e-4
        self.num_epochs = 32
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cpu")
        # self.device = torch.device("cuda:0")
        # torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.criterion = MixedLoss(10.0, 2.0)
        # self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.optimizer = RAdam(model.parameters(), lr=self.lr)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=3, verbose=True
        )
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                fold=1,
                total_folds=5,
                data_folder=data_folder,
                df_path=df_path,
                phase=phase,
                size=512,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )

            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)

        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        #         tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()

        for itr, batch in enumerate(dataloader):
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps

            if phase == "train":
                loss.backward()

                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
        #             tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = metric_get_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()

        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            val_loss = self.iterate(epoch, "val")
            self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "./model.pth")
            print()


class GPUTrainer(LearnerCallback):
    def __init__(self, learn: Learner, k: FastAIKernel):
        super().__init__(learn)
        self.k = k

    def on_train_begin(self, **kwargs: Any) -> None:
        #self.device = xm.xla_device(devkind='CPU')
        self.device = torch.device("cuda")
        self.learn.model = self.learn.model.to(self.device)
        #self.learn.data.add_tfm(partial(batch_to_device,device=self.device))
        self.old_sampler_train_dl, self.data.train_dl, self.train_sampler = _change_dl(self.k,
                                                                                       self.data.train_dl, shuffle=True)
        self.old_sampler_valid_dl, self.data.valid_dl, self.valid_sampler = _change_dl_val(self.k,
                                                                                           self.data.valid_dl, shuffle=False)

        #self.learn.data.add_tfm(partial(batch_to_device,device=self.device))
        self.learn.data.train_dl = DeviceDataLoader(
            self.data.train_dl, device=self.device)
        self.learn.data.valid_dl = DeviceDataLoader(
            self.data.valid_dl, device=self.device)
        #self.learn.data.train_dl = pl.ParallelLoader(self.data.train_dl, [self.device]).per_device_loader(self.device)
        #self.learn.data.valid_dl = pl.ParallelLoader(self.data.valid_dl, [self.device]).per_device_loader(self.device)
        #self.learn.data.train_dl.dataset = None #self.old_train_dl.dataset
        #self.learn.data.valid_dl.dataset = None #self.old_train_dl.dataset

    def on_backward_end(self, **kwargs: Any) -> None:
        #xm.optimizer_step(self.learn.opt.opt, barrier=True)
        pass
