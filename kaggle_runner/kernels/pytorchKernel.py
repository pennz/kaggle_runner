from __future__ import print_function

import collections
import functools
import os
import random
import types  # for bound new forward function for RoIHeads

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.models.detection.roi_heads as roi_heads
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.roi_heads import (fastrcnn_loss,
                                                    keypointrcnn_inference,
                                                    keypointrcnn_loss,
                                                    maskrcnn_inference,
                                                    maskrcnn_loss)
from torchvision.ops import boxes as box_ops
from torchvision.ops import roi_align
from tqdm import tqdm

import kaggle_runner.kernels.KernelRunningState
import kaggle_runner.logs
import kaggle_runner.optimizers
from kaggle_runner.kernels.kernel import KaggleKernel
from kaggle_runner.utils import kernel_utils

ImageFile.LOAD_TRUNCATED_IMAGES = True


# Plot inline
# %matplotlib inline


class PS_torch(KaggleKernel):
    def __init__(self, *args, **kargs):
        super(PS_torch, self).__init__(*args, **kargs)

        self.model_ft = None
        self.data_loader_dev = None
        self.lr_scheduler = None

        # data
        self._stat_dataset = None
        self.img_mean = [0.46877811, 0.46877811, 0.46877811]
        self.img_std = [0.24535184, 0.24535184, 0.24535184]
        # default value: mean:[0.46877811 0.46877811 0.46877811],
        # std:[0.24535184 0.24535184 0.24535184]

        # for debugging thing
        self.metric_logger = kaggle_runner.logs.MetricLogger(delimiter="  ")
        self.metric_logger.add_meter(
            "lr", kaggle_runner.logs.SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
        self.metric_logger.add_meter(
            "loss", kaggle_runner.logs.SmoothedValue(window_size=160, fmt="{avg:.6f}")
        )
        self.metric_logger.add_meter(
            "loss_mask",
            kaggle_runner.logs.SmoothedValue(window_size=160, fmt="{avg:.6f}"),
        )
        self.DATA_PATH_BASE = "../input/siimacr-pneumothorax-segmentation-data-128"

    def dump_state(
        self, exec_flag=False, force=True
    ):  # only dataloader ... others cannot dumped
        kernel_utils.logger.debug(f"state {self._stage}")
        if exec_flag:
            kernel_utils.logger.debug(f"dumping state {self._stage}")
            self_data = vars(self)

            if self.model_ft is not None:
                torch.save(self.model_ft.state_dict(), "cv_model.pth")
            names_to_exclude = {
                "model_ft",
                "optimizer",
                "lr_scheduler",
                "metric_logger",
            }

            data_to_save = {
                k: v for k, v in self_data.items() if k not in names_to_exclude
            }

            kernel_utils.dump_obj(
                data_to_save, f"run_state_{self._stage}.pkl", force=force
            )

            # print(self_data)
            # for k, v in self_data.items():
            #    utils.dump_obj(v, f'run_state_{k}_{self._stage}.pkl')

    def load_state_data_only(self, stage, file_name="run_state.pkl"):
        if stage is not None:
            file_name = f"run_state_{stage}.pkl"
        kernel_utils.logger.debug(f"restore from {file_name}")
        self_data = kernel_utils.get_obj_or_dump(filename=file_name)

        self._stage = self_data["_stage"]

        # self.model_ft = self_data['model_ft']
        self.num_epochs = self_data["num_epochs"]
        # self.optimizer = self_data['optimizer']
        self.data_loader = self_data["data_loader"]
        self.data_loader_dev = self_data["data_loader_dev"]
        self.device = self_data["device"]
        # self.lr_scheduler = self_data['lr_scheduler']

    @staticmethod
    def eval_model_loss(
        model, data_loader, device, metric_logger, print_freq, mode="train"
    ):
        # metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Evaluation"
        losses_summed = 0.0
        cnt = 0
        metric_logger.clear()

        # model.eval() # will output box, seg, scores
        if mode == "train":
            model.train()
        else:
            model.eval()  # will output box, seg, scores

        with torch.no_grad():
            for images, targets in metric_logger.log_every(
                data_loader, print_freq, header
            ):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                if mode != "train":
                    predictions, _ = model(images, targets)
                    continue
                loss_dict = model(images, targets)
                loss_dict.pop("loss_rpn_box_reg")
                # diferrent model use different names
                loss_dict.pop("loss_box_reg")

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = kernel_utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                losses_summed += losses_reduced.detach().cpu().numpy()
                cnt += 1

                metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

        metric_logger.clear()
        if mode != "train":
            return predictions
        return losses_summed / cnt

    @staticmethod
    def eval_model(model, data_loader, device, metric_logger, print_freq):
        # metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Evaluation"
        losses_reduced = float("inf")
        metric_logger.clear()

        # model.eval() # will output box, seg, scores
        with torch.no_grad():
            for images, targets in metric_logger.log_every(
                data_loader, print_freq, header
            ):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)

                losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = kernel_utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

        metric_logger.clear()
        return losses_reduced.cpu().numpy()

    @staticmethod
    def train_one_epoch(
        model,
        optimizer,
        data_loader,
        device,
        epoch,
        metric_logger,
        print_freq,
        mq_logger=None,
    ):
        model.train()
        # metric_logger = utils.MetricLogger(delimiter="  ")
        # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1,
        # fmt='{value:.6f}'))
        header = "Epoch: [{}]".format(epoch)

        metric_logger.clear()

        losses_summed = 0.0
        cnt = 0

        warm_up_lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            warm_up_lr_scheduler = kernel_utils.warmup_lr_scheduler(
                optimizer, warmup_iters, warmup_factor
            )

        for images, targets in metric_logger.log_every(data_loader, print_freq, header):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            # it just pop.. and we do not train rpn anyway!!!!
            loss_dict.pop("loss_rpn_box_reg")
            # diferrent model use different names
            loss_dict.pop("loss_box_reg")

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = kernel_utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            losses_summed += losses_reduced.detach().cpu().numpy()
            cnt += 1

            if warm_up_lr_scheduler is not None:  # only for epoch 0, warm up
                warm_up_lr_scheduler.step()

            if mq_logger is not None:  # issue is it runs off... So NAN
                mq_logger.debug(f"losses summed is {losses_summed}, cnt is {cnt}")
                print(
                    f"losses summed is {losses_summed}, cnt is {cnt}, loss_dict_reduced is {loss_dict_reduced}"
                )
            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        return losses_summed / cnt

    @staticmethod
    def _collate_fn_for_data_loader(x):
        return tuple(zip(*x))

    def prepare_train_dev_data(self):
        df = pd.read_csv(self.DATA_PATH_BASE + "/train-rle.csv")
        try:
            if self._debug_less_data:
                df = pd.read_csv(self.DATA_PATH_BASE + "/train-rle.csv")[:100]
        except Exception:
            pass

        imgdir = self.DATA_PATH_BASE + "/train/"

        if self.img_mean is None or self.img_std is None:
            self._stat_dataset = SIIMDataset_split_df(
                df, imgdir, no_aug=True
            )  # for mean calculation

        if self.submit_run:
            df_train = df
        else:
            train_mask = np.zeros((len(df),), dtype=np.bool)
            np.random.seed(2019)
            chosen = np.random.choice(len(df), int(0.8 * len(df)), replace=False)
            train_mask[chosen] = True
            val_mask = np.invert(train_mask)

            df_train = df[train_mask]
            df_dev = df[val_mask]

            dataset_dev = SIIMDataset_split_df(df_dev, imgdir, no_aug=True)
            self.data_loader_dev = torch.utils.data.DataLoader(
                dataset_dev,
                batch_size=4,
                shuffle=False,
                num_workers=4,  # 4: 08:19, 8: 08:40
                collate_fn=self._collate_fn_for_data_loader,
            )

        dataset_train = SIIMDataset_split_df(df_train, imgdir)
        # print(dataset_train[2019][1]['area'])  # only for debug
        # print(dataset_dev[19][1]['area'])

        kernel_utils.logger.debug("torch train dataloader initializing")
        self.data_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=4,
            shuffle=True,
            num_workers=4,  # 4: 08:19, 8: 08:40
            collate_fn=self._collate_fn_for_data_loader,
        )

    def after_prepare_data_hook(self):
        if self.img_mean is not None and self.img_std is not None:
            return

        stat_data_loader = torch.utils.data.DataLoader(
            self._stat_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            collate_fn=self._collate_fn_for_data_loader,
        )
        self.img_mean, self.img_std = kernel_utils.online_mean_and_sd(
            stat_data_loader, data_map=lambda x: x[0]
        )
        self.metric_logger.print_and_log_to_file(
            f"mean:{self.img_mean}, std:{self.img_std}"
        )
        del stat_data_loader

    def build_and_set_model(self):
        # create mask rcnn model
        num_classes = 2
        self.device = torch.device(
            "cpu"
        )  # TODO check if cuda is supported, or we just use cpu

        # more details at https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

        # finetuning

        # load a model pre-trained on COCO, num_classes=91, cannot change.... as the pretrained model won't load
        self.model_ft = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True, image_mean=self.img_mean, image_std=self.img_std
        )
        # FL = FocalLoss(gamma=2, alpha=0.75)  # Version 5, gamma=2, alpha=0.75, 0.8034...
        # FL = FocalLoss(gamma=1, alpha=0.75, magnifier=3)  # + early stop,2 stop at 7, (data split) version 7 0.8026
        # FL = FocalLoss(gamma=1, alpha=0.5, magnifier=3)  # version 8 0.8031, 6 epoch
        # FL = FocalLoss(gamma=0.5, alpha=0.5, magnifier=1)  # command line submission, 4 epochs cv+aug
        FL = FocalLoss(
            gamma=0.5, alpha=0.5, magnifier=1
        )  # changed lr decay 2/0.15 + patience=3, do not use focal loss...
        FL_wrapped = functools.partial(maskrcnn_loss_focal, focal_loss_func=FL)
        # FL_wrapped = None  # changed lr decay 2/0.15, do not use focal loss... 0.8025

        RoIHeads_loss_customized.set_customized_loss(
            self.model_ft.roi_heads, maskrcnn_loss_customized=FL_wrapped
        )
        RoIHeads_loss_customized.update_forward_func(self.model_ft.roi_heads)

        # get number of input features for the classifier
        in_features = self.model_ft.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model_ft.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

        # change mask prediction head, only predict background and pneu... part
        in_features_mask = self.model_ft.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model_ft.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

        # GPU
        # my_trace()  # test about to
        self.model_ft.to(self.device)
        #  self.logger.debug(f"model info:\n{self.model_ft}")

        # for param in self.model_ft.parameters():
        #    param.requires_grad = True

        params = [p for p in self.model_ft.parameters() if p.requires_grad]

        start_learning_rate = 0.001

        try:
            if self._debug_continue_training:  # monkey patch
                start_learning_rate = 0.00001
        except Exception:
            pass
        self.optimizer = torch.optim.SGD(
            params, lr=start_learning_rate, momentum=0.9, weight_decay=0.0005
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=4,
            # after changed to 3, 0.8042 (improved from 0.8033)
            # step_size 4, with little aug, 0.8037 (Version 11)
            gamma=0.1,
        )

    def train_model(self):
        patience = 3
        if self.submit_run:
            patience = 0
        es = kaggle_runner.optimizers.EarlyStopping(
            patience=patience
        )  # the first time it become worse, if patience set to 1

        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch(
                self.model_ft,
                self.optimizer,
                self.data_loader,
                self.device,
                epoch,
                self.metric_logger,
                print_freq=10,
                mq_logger=self.logger,
            )
            self.metric_logger.print_and_log_to_file(
                f"train_loss (averaged) is {train_loss}"
            )
            self.lr_scheduler.step()  # change learning rate

            if not self.submit_run:
                metric = self.eval_model_loss(
                    self.model_ft,
                    self.data_loader_dev,
                    self.device,
                    self.metric_logger,
                    print_freq=10,
                )
                self.metric_logger.print_and_log_to_file(
                    f"\nmetric (averaged) is {metric}\n"
                )
                if es.step(metric):
                    self.print_log(
                        f"{epoch+1} epochs run and early stop, with patience {patience}"
                    )
                    break

    def print_log(self, s):
        self.metric_logger.print_and_log_to_file(s)

    def save_model(self):
        torch.save(self.model_ft.state_dict(), "cv_model.pth")

    def load_model_weight_continue_train(self):
        self.build_and_set_model()
        assert os.path.exists("cv_model.pth")
        self.model_ft.load_state_dict(torch.load("cv_model.pth"))
        self.model_ft.eval()

        patience = 3
        if self.submit_run:
            patience = 0
        es = kaggle_runner.optimizers.EarlyStopping(
            patience=patience
        )  # the first time it become worse, if patience set to 1
        for epoch in range(8, 13):
            train_loss = self.train_one_epoch(
                self.model_ft,
                self.optimizer,
                self.data_loader,
                self.device,
                epoch,
                self.metric_logger,
                print_freq=100,
            )
            print(f"train_loss (averaged) is {train_loss}")
            self.lr_scheduler.step()  # change learning rate

            if not self.submit_run:
                metric = self.eval_model_loss(
                    self.model_ft,
                    self.data_loader_dev,
                    self.device,
                    self.metric_logger,
                    print_freq=100,
                )
                print(f"metric (averaged) is {metric}")
                if es.step(metric):
                    print(
                        f"{epoch+1} epochs run and early stop, with patience {patience}"
                    )
                    break

    def load_model_weight(self):
        self.build_and_set_model()
        self.model_ft.load_state_dict(torch.load("cv_model.pth"))
        self.model_ft.eval()

    def pre_test(self):
        # change model to prediction mode
        for param in self.model_ft.parameters():
            param.requires_grad = False

        self.model_ft.eval()

    def predict_rle_from_acts_with_threshold(self, acts, threshold):
        return self.predict_rle_from_acts(
            acts, self.data_loader_dev, self.metric_logger, 100, threshold
        )

    @staticmethod
    def predict_rle_from_acts(
        acts, data_loader, metric_logger, print_freq, threshold=0.5
    ):
        header = "DEV prediction test"

        sublist = []
        counter = 0

        data_loader_b1 = torch.utils.data.DataLoader(
            data_loader.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # 4: 08:19, 8: 08:40
            collate_fn=PS_torch._collate_fn_for_data_loader,
        )

        idx = 0
        mask_add_cnt = []
        for image, target in metric_logger.log_every(
            data_loader_b1, print_freq, header
        ):
            width, height = 1024, 1024

            result = acts[idx]
            image_id = target[0]["image_id"].cpu().numpy()
            idx += 1
            if len(result["masks"]) > 0:
                counter += 1
                mask_added = 0
                for ppx in range(len(result["masks"])):
                    if result["scores"][ppx] >= threshold:
                        mask_added += 1
                        # res = transforms.ToPILImage()(result["masks"][ppx].permute(1, 2, 0).cpu().numpy())
                        res = transforms.ToPILImage()(
                            result["masks"][ppx].transpose((1, 2, 0))
                        )
                        res = np.asarray(
                            res.resize((width, height), resample=Image.BILINEAR)
                        )
                        res = (res[:, :] * 255.0 > 127).astype(np.uint8).T
                        rle = kernel_utils.mask_to_rle(res, width, height)
                        sublist.append([image_id, rle])
                if mask_added == 0:
                    rle = " -1"
                    sublist.append([image_id, rle])
                mask_add_cnt.append(mask_added)
            else:
                rle = " -1"
                sublist.append([image_id, rle])
                mask_add_cnt.append(0)  # no mask
            if idx % 100 == 0:
                mask_stat = np.array(mask_add_cnt)
                metric_logger.update(
                    **{
                        "mask0": ((mask_stat == 0).sum() / idx),
                        "mask1": ((mask_stat == 1).sum() / idx),
                        "mask2": ((mask_stat == 2).sum() / idx),
                        "mask>2": ((mask_stat > 2).sum() / idx),
                    }
                )

        mask_stat = np.array(mask_add_cnt)
        assert len(mask_stat) == idx

        metric_logger.print_and_log_to_file(
            f"image cnt: {idx}, image predicted mask cnt: {counter}, "
            f"mask 0,1,2,3+ {(mask_stat==0).sum()/idx}, "
            f"{(mask_stat==1).sum()/idx}, {(mask_stat==2).sum()/idx}, {(mask_stat>2).sum()/idx}"
        )

        return sublist, mask_stat

    @staticmethod
    def predict_rle(model, device, data_loader, metric_logger, print_freq):
        header = "DEV prediction test"

        sublist = []
        counter = 0
        # changed from 0.25 to 0.5... need to check the data and analyze...(test on dev set)
        threshold = 0.55
        assert data_loader.batch_size == 1

        for images, targets in metric_logger.log_every(data_loader, print_freq, header):
            width, height = images.size
            image_id = targets["image_id"]

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            result = model(images, targets)[0]

            if len(result["masks"]) > 0:
                counter += 1
                mask_added = 0
                for ppx in range(len(result["masks"])):
                    if result["scores"][ppx] >= threshold:
                        mask_added += 1
                        res = transforms.ToPILImage()(
                            result["masks"][ppx].permute(1, 2, 0).cpu().numpy()
                        )
                        res = np.asarray(
                            res.resize((width, height), resample=Image.BILINEAR)
                        )
                        res = (res[:, :] * 255.0 > 127).astype(np.uint8).T
                        rle = kernel_utils.mask_to_rle(res, width, height)
                        sublist.append([image_id, rle])
                if mask_added == 0:
                    rle = " -1"
                    sublist.append([image_id, rle])
            else:
                rle = " -1"
                sublist.append([image_id, rle])

        return sublist

    def predict_on_test(self):
        sample_df = pd.read_csv(
            os.path.join(self.DATA_PATH_BASE, "sample_submission.csv")
        )

        # this part was taken from @raddar's kernel: https://www.kaggle.com/raddar/better-sample-submission
        masks_ = sample_df.groupby("ImageId")["ImageId"].count().reset_index(name="N")
        masks_ = masks_.loc[masks_.N > 1].ImageId.values
        ###
        sample_df = sample_df.drop_duplicates("ImageId", keep="last").reset_index(
            drop=True
        )

        tt = transforms.ToTensor()
        sublist = []
        counter = 0
        # changed from 0.25 to 0.5... need to check the data and analyze...(test on dev set)
        threshold = 0.55
        for index, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
            image_id = row["ImageId"]
            if image_id in masks_:
                img_path = os.path.join(
                    self.DATA_PATH_BASE + "/test/" + image_id + ".png"
                )

                img = Image.open(img_path).convert("RGB")
                width, height = img.size
                img = img.resize((1024, 1024), resample=Image.BILINEAR)
                img = tt(img)
                result = self.model_ft([img.to(self.device)])[0]
                if len(result["masks"]) > 0:
                    counter += 1
                    mask_added = 0
                    for ppx in range(len(result["masks"])):
                        if result["scores"][ppx] >= threshold:
                            mask_added += 1
                            res = transforms.ToPILImage()(
                                result["masks"][ppx].permute(1, 2, 0).cpu().numpy()
                            )
                            res = np.asarray(
                                res.resize((width, height), resample=Image.BILINEAR)
                            )
                            res = (res[:, :] * 255.0 > 127).astype(np.uint8).T
                            rle = kernel_utils.mask_to_rle(res, width, height)
                            sublist.append([image_id, rle])
                    if mask_added == 0:
                        rle = " -1"
                        sublist.append([image_id, rle])
                else:
                    rle = " -1"
                    sublist.append([image_id, rle])
            else:
                rle = " -1"
                sublist.append([image_id, rle])

        submission_df = pd.DataFrame(sublist, columns=sample_df.columns.values)
        submission_df.to_csv("submission.csv", index=False)
        print(counter)

    def _build_show_model_detail(self):
        self.run(
            end_stage=kaggle_runner.kernels.KernelRunningState.KernelRunningState.PREPARE_DATA_DONE
        )
        self.build_and_set_model()
        # self.pre_test()
        # summary(self.model_ft, (1,1024,1024))

    def check_predict_details(self):
        # Thanks https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6
        assert self.model_ft is not None
        activations = kernel_utils.get_obj_or_dump("dev_output_results.pkl")
        self.analyzer = TorchModelAnalyzer(self)
        analyzer = self.analyzer

        if activations is not None:
            self.analyzer.activation = activations
        else:
            analyzer.register_forward_hook(
                self.model_ft.roi_heads, analyzer.get_output_saved("roi_heads")
            )
            # for mask_head output, size torch.Size([40, 256, 14, 14]),
            # (mask_fcn4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # (relu4): ReLU(inplace)
            # for mask_predictor, torch.Size([46, 2, 28, 28])  (in evaluate mode)
            # for roi_heads, output.shape, tuple length 2, ([{'boxes', 'labels', 'scores':torch.Size([7]), 'masks':torch.Size([7, 1, 28, 28])},T,T,T],[ PLACE_FOR_LOSSES ])
            # for roi_heads, in training mode, output different number of test result, according to threshold thing

            self.eval_model_loss(
                self.model_ft,
                self.data_loader_dev,
                self.device,
                self.metric_logger,
                print_freq=150,
            )
            kernel_utils.dump_obj(
                analyzer.activation, "dev_output_results.pkl", force=True
            )

        roi_acts = []
        for acts in analyzer.activation["roi_heads"]:
            roi_acts += acts[0]
        self.analyzer.test_out_threshold(roi_acts)
        # my_trace()


class SIIMDataset(torch.utils.data.Dataset):
    def __init__(self, df_path, img_dir):
        self.df = pd.read_csv(df_path)
        self.height = 1024
        self.width = 1024
        self.image_dir = img_dir
        self.image_info = collections.defaultdict(dict)
        self.to_tensor_transformer = transforms.ToTensor()

        self.load_image_info()

    def load_image_info(self):
        counter = 0
        for index, row in tqdm(self.df.iterrows(), total=len(self.df)):
            image_id = row["ImageId"]
            image_path = os.path.join(self.image_dir, image_id)
            if (
                os.path.exists(image_path + ".png")
                and row[" EncodedPixels"].strip() != "-1"
            ):
                self.image_info[counter]["image_id"] = image_id
                self.image_info[counter]["image_path"] = image_path
                self.image_info[counter]["annotations"] = row[" EncodedPixels"].strip()
                counter += 1

    def _test_(self, idx):
        self.__getitem__(idx)

    def __getitem__(self, idx):
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path + ".png").convert(
            "RGB"
        )  # here it is converted to 3 channels
        width, height = img.size
        img = img.resize((self.width, self.height), resample=Image.BILINEAR)
        info = self.image_info[idx]

        mask = kernel_utils.rle2mask(info["annotations"], width, height)

        if mask is None:
            raise ValueError("mask is None!!!!!!!!")

        mask = Image.fromarray(mask.T)
        mask = mask.resize((self.width, self.height), resample=Image.BILINEAR)
        img, mask = self.functional_transforms(
            img, mask
        )  # pay attention need after mask.T
        mask = np.expand_dims(mask, axis=0)

        if sum(mask.flatten()) == 0:
            xmax = 0
            xmin = 0
            ymax = 0
            ymin = 0
        else:
            pos = np.where(np.array(mask)[0, :, :])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

        boxes = torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
        labels = torch.ones((1,), dtype=torch.int64)
        masks = torch.as_tensor(mask, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        img = self.to_tensor_transformer(img)
        return img, target

    def functional_transforms(self, img, mask):
        # transforms.compose([
        #    transforms.ColorJitter(brightness=0.3, contrast=0.3),
        #    transforms.RandomAffine(degrees=15, translate=(0.1, 0.2)),
        #    transforms.RandomHorizontalFlip(p=0.5),
        #    transforms.RandomResizedCrop((self.width, self.height), scale=(0.2, 1.2), ratio=(0.9,1.1111))
        # ])
        pos_cnt = np.array(mask).sum()
        image = img
        segmentation = mask

        if random.random() > 0.5:
            while True:
                angle = random.uniform(-10, 10)
                translate_x = int(random.uniform(-0.05, 0.05) * self.width)
                translate_y = int(random.uniform(-0.1, 0.1) * self.height)
                scale = random.uniform(0.8, 1.2)

                image = TF.affine(
                    img, angle, (translate_x, translate_y), scale, shear=0
                )
                segmentation = TF.affine(
                    mask, angle, (translate_x, translate_y), scale, shear=0
                )
                if pos_cnt > 0:
                    trans_pos_cnt = np.array(segmentation).sum()
                    if trans_pos_cnt / (pos_cnt * scale) < 0.95:
                        continue  # not move the mask out

                # no more translation
                # angle = random.randint(-15, 15)
                # image = TF.rotate(img, angle)
                # segmentation = TF.rotate(mask, angle)
                # more transforms ...
                # bright_f = random.uniform(-0.3, 0.3)
                # image = TF.adjust_brightness(image, bright_f)
                ##segmentation = TF.adjust_brightness(segmentation, bright_f)

                # contrast_f = random.uniform(-0.3, 0.3)
                # image = TF.adjust_contrast(image, bright_f)
                ##segmentation = TF.adjust_contrast(segmentation, bright_f)
                # image = TF.resize(image, (self.height, self.width))
                # segmentation = TF.resize(segmentation, (self.height, self.width))  # the network can accept different size

                gamma = random.uniform(0.75, 1.3333)
                image = TF.adjust_gamma(image, gamma)
                break
        if random.random() > 0.5:
            image = TF.hflip(image)
            segmentation = TF.hflip(segmentation)

        return image, segmentation

    def __len__(self):
        return len(self.image_info)


class SIIMDataset_split_df(SIIMDataset):
    def __init__(self, df, img_dir, no_aug=False):
        # super(SIIMDataset_split_df, self).__init__()
        self.df = df
        self.height = 1024
        self.width = 1024
        self.image_dir = img_dir
        self.image_info = collections.defaultdict(dict)

        self.no_aug = no_aug

        self.to_tensor_transformer = transforms.ToTensor()
        self.load_image_info()

    def functional_transforms(self, img, mask):
        if self.no_aug:
            return img, mask
        else:
            return super(SIIMDataset_split_df, self).functional_transforms(img, mask)


# modified based https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=0.5, eps=1e-7, magnifier=1.0, from_logits=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.magnifier = magnifier
        assert self.alpha >= 0
        assert self.alpha <= 1
        assert self.magnifier > 0
        self.from_logits = from_logits

    def forward(self, input, target):
        if self.from_logits:
            input = torch.sigmoid(input)

        y = target
        not_y = 1 - target

        y_hat = input
        not_y_hat = 1 - input

        y_hat = y_hat.clamp(self.eps, 1.0 - self.eps)
        not_y_hat = not_y_hat.clamp(self.eps, 1.0 - self.eps)

        loss = (
            -1 * self.alpha * not_y_hat ** self.gamma * y * torch.log(y_hat)
        )  # cross entropy
        loss += (
            -1 * (1 - self.alpha) * y_hat ** self.gamma * not_y * torch.log(not_y_hat)
        )
        loss *= self.magnifier

        return loss.mean()


class RoIHeads_loss_customized(roi_heads.RoIHeads):
    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
        maskrcnn_loss_customized=None,
        fastrcnn_loss_customized=None,
        keypointrcnn_loss_customized=None,
    ):
        super(RoIHeads_loss_customized, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

        self.keypoint_roi_pool = keypoint_roi_pool
        self.keypoint_head = keypoint_head
        self.keypoint_predictor = keypoint_predictor

        self.maskrcnn_loss_customized = maskrcnn_loss_customized
        self.fastrcnn_loss_customized = fastrcnn_loss_customized
        self.keypointrcnn_loss_customized = keypointrcnn_loss_customized

    @staticmethod
    def set_customized_loss(
        head,
        maskrcnn_loss_customized=None,
        fastrcnn_loss_customized=None,
        keypointrcnn_loss_customized=None,
    ):
        head.maskrcnn_loss_customized = maskrcnn_loss_customized
        head.fastrcnn_loss_customized = fastrcnn_loss_customized
        head.keypointrcnn_loss_customized = keypointrcnn_loss_customized

    @staticmethod
    def update_forward_func(head):
        head.forward = types.MethodType(
            RoIHeads_loss_customized.forward, head
        )  # bound the method to our head

    # def forward(self, features, proposals, image_shapes, targets=None):
    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        maskrcnn_loss_func = maskrcnn_loss
        fastrcnn_loss_func = fastrcnn_loss
        keypointrcnn_loss_func = keypointrcnn_loss

        eval_when_train = not self.training
        try:
            if self._eval_when_train:
                eval_when_train = True
        except AttributeError:
            pass

        if self.maskrcnn_loss_customized is not None:
            maskrcnn_loss_func = self.maskrcnn_loss_customized
        if self.fastrcnn_loss_customized is not None:
            fastrcnn_loss_func = self.fastrcnn_loss_customized
        if self.keypointrcnn_loss_customized is not None:
            keypointrcnn_loss_func = self.keypointrcnn_loss_customized

        if self.training:
            (
                proposals,
                matched_idxs,
                labels,
                regression_targets,
            ) = self.select_training_samples(proposals, targets)

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result, losses = [], {}
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss_func(
                class_logits, box_regression, labels, regression_targets
            )
            losses = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)
        if eval_when_train:
            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes
            )
            num_images = len(boxes)
            for i in range(num_images):
                result.append(dict(boxes=boxes[i], labels=labels[i], scores=scores[i],))

        if self.has_mask:
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])

            mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)

            loss_mask = {}
            if self.training:
                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                loss_mask = maskrcnn_loss_func(
                    mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs
                )
                loss_mask = dict(loss_mask=loss_mask)
            if eval_when_train:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        if self.has_keypoint():
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])

            keypoint_features = self.keypoint_roi_pool(
                features, keypoint_proposals, image_shapes
            )
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                gt_keypoints = [t["keypoints"] for t in targets]
                loss_keypoint = keypointrcnn_loss_func(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = dict(loss_keypoint=loss_keypoint)
            if eval_when_train:
                keypoints_probs, kp_scores = keypointrcnn_inference(
                    keypoint_logits, keypoint_proposals
                )
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses


def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)
    return roi_align(gt_masks, rois, (M, M), 1)[:, 0]


def maskrcnn_loss_focal(
    mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs, focal_loss_func=None
):
    """
    Arguments:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    discretization_size = mask_logits.shape[-1]
    labels = [l[idxs] for l, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    loss_func = F.binary_cross_entropy_with_logits
    if focal_loss_func is not None:
        loss_func = focal_loss_func

    mask_loss = loss_func(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels],
        mask_targets,
    )
    return mask_loss


class TorchModelAnalyzer:
    def __init__(self, kernel):
        self.activation = {}
        self.kernel = kernel

    def get_output_saved(self, name):
        self.activation[name] = []

        def rpn_output_hook(model, input, output):
            if name == "roi_heads":
                self.activation[name].append(self.roi_heads_output_detach(output))

        return rpn_output_hook

    # def put_one_predict(self, name, detached_output):

    def print_output(self, name):
        print(self.activation.get(name, f"output information for {name} not found"))

    def test_out_threshold(self, activations):
        stat_for_threshold = {}
        pred_for_threshold = {}
        for threshold in [0.5, 0.52, 0.55, 0.6]:
            preds, stat = self.kernel.predict_rle_from_acts_with_threshold(
                activations, threshold
            )

            stat_for_threshold[threshold] = stat
            pred_for_threshold[threshold] = preds
            # my_trace()
            # print(self.metric_cal(preds))

        kernel_utils.dump_obj(stat_for_threshold, "stat_for_threshold.pkl", force=True)
        kernel_utils.dump_obj(pred_for_threshold, "pred_for_threshold.pkl", force=True)

    def metric_cal(self, preds):
        pass

    @staticmethod
    def roi_heads_output_detach(output):
        return (
            [
                {k: v.detach().cpu().numpy() for k, v in dict.items()}
                for dict in output[0]
            ],
            {k: v.detach().cpu().numpy() for k, v in output[1].items()},
        )

    @staticmethod
    def register_forward_hook(module, func_to_hook):
        module.register_forward_hook(func_to_hook)
