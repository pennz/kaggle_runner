import gc
import importlib
import os
import random
import unittest

import numpy as np
import pytest
import torch
from torch.autograd import Variable

# import modelTester
from kaggle_runner.runners import runner
from kaggle_runner.data_providers import provider
from kaggle_runner.kernels import PSKernel, kernel, pytorchKernel
from kaggle_runner.kernels.KernelRunningState import KernelRunningState
# from kernel import Kernel
from kaggle_runner.utils import kernel_utils

log_args = {
    "size": 384,
    "network": "intercept",
    "AMQPURL": "amqp://drdsfaew:QrBHPPxbsd8IuIxKrCnX3-RGoLKaFhYI@termite.rmq.cloudamqp.com/drdsfaew",
    "seed": 19999,
}
r = runner.Runner(
    log_args["network"],
    log_args["AMQPURL"],
    size=log_args["size"],
    seed=log_args["seed"],
)
r._attach_data_collector("")
LOGGER = r.logger


@pytest.fixture(scope="session")
def args():
    return {
        "size": 384,
        "network": "intercept",
        "AMQPURL": "amqp://drdsfaew:QrBHPPxbsd8IuIxKrCnX3-RGoLKaFhYI@termite.rmq.cloudamqp.com/drdsfaew",
        "seed": 19999,
    }


@pytest.fixture(scope="session")
def mq_logger():
    log_args = {
        "size": 384,
        "network": "intercept",
        "AMQPURL": "amqp://drdsfaew:QrBHPPxbsd8IuIxKrCnX3-RGoLKaFhYI@termite.rmq.cloudamqp.com/drdsfaew",
        "seed": 19999,
    }
    r = runner.Runner(
        log_args["network"],
        log_args["AMQPURL"],
        size=log_args["size"],
        seed=log_args["seed"],
    )
    r._attach_data_collector("")
    _mq_logger = r.logger
    return _mq_logger


class TestModelBasic:
    ModelList = []

    def test_creation(self):
        pass

    def test_load_model(self):
        pass

    def test_eval(self):
        pass

    def test_dataloaders(self):
        """test_dataloaders uses codes from unet-with-se-resnet ..., it uses
        provider to generate dataloader"""
        train_rle_path = "../input/mysiim/train-rle.csv"
        data_folder = "../input/siimpng/siimpng/train_png"
        dataloader = provider(
            fold=0,
            total_folds=5,
            data_folder=data_folder,
            df_path=train_rle_path,
            phase="train",
            size=512,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            batch_size=16,
            num_workers=2,
        )

        batch = next(iter(dataloader))  # get a batch from the dataloader
        images, masks = batch

        # plot some random images in the `batch`
        idx = random.choice(range(16))
        # plt.imshow(images[idx][0], cmap="bone")
        # plt.imshow(masks[idx][0], alpha=0.2, cmap="Reds")
        # plt.show()
        if len(np.unique(masks[idx][0])) == 1:  # only zeros
            print("Chosen image has no ground truth mask, rerun the cell")


class TestMQLogger:
    def test_runner_create(self, args):
        r = runner.Runner(
            args["network"], args["AMQPURL"], size=args["size"], seed=args["seed"]
        )
        assert r.AMQPURL is not None

    def test_runner_logger_use_elsewhere(self, args):
        r = runner.Runner(
            args["network"], args["AMQPURL"], size=args["size"], seed=args["seed"]
        )
        assert r.AMQPURL is not None
        r._attach_data_collector("")
        assert r.logger is not None
        r.logger.debug("use elsewhere")


@pytest.yield_fixture(scope="session")
def session_thing(mq_logger):
    mq_logger.debug("constructing session thing")
    yield
    mq_logger.debug("destroying session thing")


@pytest.yield_fixture
def testcase_thing(mq_logger):
    mq_logger.debug("constructing testcase thing")
    yield
    mq_logger.debug("destroying testcase thing")


class TestPSKernel:
    # this function will run before every test. We re-initialize group in this
    # function. So for every test, new group is used.
    @classmethod
    def setup_class(cls):
        gc.enable()
        importlib.reload(kernel_utils)
        importlib.reload(kernel)
        importlib.reload(PSKernel)
        importlib.reload(pytorchKernel)
        # importlib.reload(modelTester)
        log_args = {
            "size": 384,
            "network": "intercept",
            "AMQPURL": "amqp://drdsfaew:QrBHPPxbsd8IuIxKrCnX3-RGoLKaFhYI@termite.rmq.cloudamqp.com/drdsfaew",
            "seed": 19999,
        }
        r = runner.Runner(
            log_args["network"],
            log_args["AMQPURL"],
            size=log_args["size"],
            seed=log_args["seed"],
        )
        r._attach_data_collector("")
        cls.logger = r.logger
        cls.logger.debug("Good day~")

    @classmethod
    def teardown_class(cls):
        cls.logger.debug("Keep happy~")

    def setup_method(self, method):
        self.logger.debug("setup for method %s", method)

    def teardown_method(self, method):
        self.logger.debug("teardown method %s", method)
        gc.collect()

    def test_class(self, mq_logger):
        ps_kernel = PSKernel.PS(mq_logger)
        # set_trace()
        assert len(ps_kernel.model_metrics) == 0

    @pytest.mark.skip("take too long to test, just skip")
    def test_prepare_data(self, mq_logger):
        ps_kernel = PSKernel.PS(mq_logger)
        ps_kernel.run(
            end_stage=KernelRunningState.PREPARE_DATA_DONE, dump_flag=False
        )  # will also analyze data
        assert ps_kernel.train_X is not None
        assert len(ps_kernel.train_X) == len(ps_kernel.train_Y)
        # self.assertIsNotNone(ps_kernel.test_X)  // don't care this now
        assert ps_kernel.dev_X is not None
        assert len(ps_kernel.dev_X) == len(ps_kernel.dev_Y)

    @pytest.mark.skip("take too long to test, just skip")
    def test_dump_load_continue(self, mq_logger):
        ps_kernel = PSKernel.PS(mq_logger)
        ps_kernel.run(end_stage=KernelRunningState.TRAINING_DONE)
        assert ps_kernel._stage == KernelRunningState.TRAINING_DONE

        kernel_load_back = kernel.KaggleKernel._load_state(logger=mq_logger)
        assert kernel_load_back._stage == KernelRunningState.TRAINING_DONE
        kernel_load_back.run()
        assert kernel_load_back._stage == KernelRunningState.SAVE_SUBMISSION_DONE

    @pytest.mark.skip("take too long to test, just skip")
    def test_train(self, mq_logger):
        # kernel_load_back = kernel.KaggleKernel._load_state(
        #     KernelRunningState.PREPARE_DATA_DONE, logger=mq_logger
        # )
        ps_kernel = PSKernel.PS(mq_logger)
        ps_kernel.run(end_stage=KernelRunningState.TRAINING_DONE)
        assert ps_kernel.model is not None

    @pytest.mark.skip("take too long to test, just skip")
    def test_read_tf(self, mq_logger):
        k = PSKernel.PS(mq_logger)
        k._recover_from_tf()
        # k.run(start_stage=KernelRunningState.PREPARE_DATA_DONE,
        # end_stage=KernelRunningState.TRAINING_DONE)
        assert k.ds is not None

    @pytest.mark.skip("take too long to test, just skip")
    def test_convert_tf_from_start(self, mq_logger):  # won't work
        ps_kernel = PSKernel.PS(mq_logger)
        ps_kernel.run(end_stage=KernelRunningState.PREPARE_DATA_DONE)
        assert os.path.isfile("train_dev.10.tfrec")

    # def test_convert_tf(self, mq_logger):
    #    kernel_withdata
    #    = kernel.KaggleKernel._load_state(KernelRunningState.PREPARE_DATA_DONE)
    #    k = PSKernel.PS(mq_logger)
    #    k._clone_data(kernel_withdata)
    #    k.after_prepare_data_hook()

    #    self.assertTrue(os.path.isfile('train.tfrec'))


@pytest.mark.skip("take too long to test, just skip")
class TestPytorchKernel:
    @classmethod
    def setup_class(cls):
        gc.enable()
        importlib.reload(kernel_utils)
        importlib.reload(kernel)
        importlib.reload(PSKernel)
        importlib.reload(pytorchKernel)
        # importlib.reload(modelTester)
        log_args = {
            "size": 384,
            "network": "intercept",
            "AMQPURL": "amqp://drdsfaew:QrBHPPxbsd8IuIxKrCnX3-RGoLKaFhYI@termite.rmq.cloudamqp.com/drdsfaew",
            "seed": 19999,
        }
        r = runner.Runner(
            log_args["network"],
            log_args["AMQPURL"],
            size=log_args["size"],
            seed=log_args["seed"],
        )
        r._attach_data_collector("")
        cls.logger = r.logger

        cls.logger.debug("Good day~")

    @classmethod
    def teardown_class(cls):
        cls.logger.debug("Keep happy~")

    def setup_method(self, method):
        self.logger.debug("setup for method %s", method)

    def teardown_method(self, method):
        self.logger.debug("teardown method %s", method)
        gc.collect()

    def test_pytorch_data_aug(self, mq_logger):
        self._prepare_data(mq_logger)

        k = pytorchKernel.PS_torch(mq_logger)
        k._debug_less_data = True
        k.run(end_stage=KernelRunningState.PREPARE_DATA_DONE)
        # k.load_state_data_only(KernelRunningState.PREPARE_DATA_DONE)
        # k.run(end_stage=KernelRunningState.PREPARE_DATA_DONE,
        # dump_flag=True)  # will also analyze data
        k.data_loader.dataset._test_(1)
        k.data_loader.dataset._test_(2)  # test should choose small idx, as
        # batch might be small
        # k.run()  # dump not working for torch
        assert k is not None

    def _prepare_data(self, mq_logger):
        _stage = KernelRunningState.PREPARE_DATA_DONE
        data_stage_file_name = "run_state_%s.pkl" % _stage
        if not os.path.isfile(data_stage_file_name):
            self._pytorch_starter_dump(mq_logger)

    @pytest.mark.skip("data dump not avaliable")
    def test_pytorch_FL_in_model_early_stop(self, mq_logger):
        self._prepare_data(mq_logger)
        kernel_load_back = pytorchKernel.PS_torch(mq_logger)

        kernel_load_back.load_state_data_only(
            KernelRunningState.PREPARE_DATA_DONE)
        kernel_load_back.build_and_set_model()
        kernel_load_back.train_model()

    @pytest.mark.skip("we train more")
    def test_pytorch_cv_train_dev(self, mq_logger):
        k = pytorchKernel.PS_torch(mq_logger)

        k._debug_less_data = True
        k.run(end_stage=KernelRunningState.PREPARE_DATA_DONE)
        k.logger.debug("data done")

        k.build_and_set_model()
        k.num_epochs = 1
        k.logger.debug("start train one epoch")
        k.train_model()
        k.logger.debug("end train one epoch")
        assert True

    def test_pytorch_cv_train_more(self, mq_logger):
        k = pytorchKernel.PS_torch(mq_logger)

        # k._debug_less_data = True
        k.run(end_stage=KernelRunningState.PREPARE_DATA_DONE)

        k.build_and_set_model()
        k.logger.debug("start train 5 epochs")
        k.num_epochs = 5
        k.train_model()

    def test_pytorch_cv_data_prepare(self, mq_logger):
        k = pytorchKernel.PS_torch(mq_logger)
        k.run(end_stage=KernelRunningState.PREPARE_DATA_DONE)

        # k2 = pytorchKernel.PS_torch(mq_logger)
        # k2.run(end_stage=KernelRunningState.PREPARE_DATA_DONE)

        l = len(k.data_loader)
        l_dev = len(k.data_loader_dev)
        ratio = l / l_dev

        assert ratio > 3.5  # around 0.8/0.2
        assert ratio < 4.5

    def test_pytorch_focal_loss_func(self, mq_logger):
        inputs = torch.tensor(
            [
                [0.0, 0.2, 0.4, 0.0, 0.2, 0.4, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                [0.0, 0.2, 0.4, 0.0, 0.2, 0.4, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ]
        )
        targets = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ]
        )

        FL = pytorchKernel.FocalLoss(gamma=2)
        FL_normal_CE = pytorchKernel.FocalLoss(gamma=0)
        alpha_for_pos = 0.75
        FL_alpha_balance = pytorchKernel.FocalLoss(
            gamma=0, alpha=alpha_for_pos
        )  # alpha for positive class weights

        print("----inputs----")
        print(inputs)
        print("---target-----")
        print(targets)
        losses = []
        grads = []

        for loss_func in [FL, FL_normal_CE, FL_alpha_balance, FL_normal_CE]:
            inputs_fl = Variable(inputs.clone(), requires_grad=True)
            targets_fl = Variable(targets.clone())
            fl_loss = loss_func(inputs_fl, targets_fl)
            print(fl_loss.data)
            fl_loss.backward()
            print(inputs_fl.grad.data)
            losses.append(fl_loss.data.numpy())
            grads.append(inputs_fl.grad.data.numpy())
        assert losses[-1] == losses[1]
        if alpha_for_pos >= 0.5:
            assert losses[2] < losses[1]
        else:
            assert losses[2] > losses[1]

        assert (grads[2][0, 1] / grads[1][0, 1]) == (1 - alpha_for_pos) / 0.5

    @pytest.mark.skip("won't work for pytorch")
    def _pytorch_starter_dump(self, mq_logger):
        k = pytorchKernel.PS_torch(mq_logger)
        k.run(
            end_stage=KernelRunningState.PREPARE_DATA_DONE, dump_flag=True
        )  # will also analyze data
        # kernel_load_back = pytorchKernel.PS_torch(mq_logger)

        # kernel_load_back.load_state_data_only(KernelRunningState.PREPARE_DATA_DONE)
        # kernel_load_back.run(end_stage=KernelRunningState.TRAINING_DONE)

    @pytest.mark.skip("won't work for pytorch")
    def test_pytorch_starter_load(self, mq_logger):
        kernel_load_back = pytorchKernel.PS_torch(mq_logger)

        kernel_load_back.load_state_data_only(KernelRunningState.TRAINING_DONE)
        kernel_load_back.load_model_weight()

    @pytest.mark.skip("won't work for pytorch")
    def test_pytorch_starter_load_then_train(self, mq_logger):
        k = pytorchKernel.PS_torch(mq_logger)

        # will also analyze data
        k.run(end_stage=KernelRunningState.PREPARE_DATA_DONE)
        # k._debug_continue_training = True
        k.load_model_weight_continue_train()
        # kernel_load_back.run(end_stage=KernelRunningState.TRAINING_DONE)

    @pytest.mark.skip("won't work for pytorch")
    def test_pytorch_starter_load_then_submit(self, mq_logger):
        kernel_load_back = pytorchKernel.PS_torch(mq_logger)

        kernel_load_back.load_state_data_only(KernelRunningState.TRAINING_DONE)
        kernel_load_back.load_model_weight()
        kernel_load_back.run()
        # kernel_load_back.run(end_stage=KernelRunningState.TRAINING_DONE)

    def test_pytorch_dataset_mean_std(self, mq_logger):
        k = pytorchKernel.PS_torch(mq_logger)
        # k.run(end_stage=KernelRunningState.PREPARE_DATA_DONE,
        # dump_flag=True)  # will also analyze data
        k._debug_less_data = True
        k.run(
            end_stage=KernelRunningState.PREPARE_DATA_DONE, dump_flag=False
        )  # dump not working for torch
        k.pre_train()
        assert k.img_mean is not None

    # def test_tf_model_zoo(self, mq_logger):
    #     t = modelTester.TF_model_zoo_tester()
    #     t.run_logic()

    # def test_tf_model_zoo_model(self, mq_logger):
    #     t = modelTester.TF_model_zoo_tester()
    #     t.load_model()
    #     assert t.model is not None

    # def test_tf_model_zoo_model(self, mq_logger):
    #     t = modelTester.TF_model_zoo_tester()
    #     t.set_model("mask_rcnn_resnet101_atrous_coco_2018_01_28")
    #     t.run_prepare()
    #     # t.check_graph()
    #     t.run_test()  # result is great!!!
    #     assert t.detection_graph is not None

    @pytest.mark.skip()
    def test_analyze_RPN(self, mq_logger):
        assert False

    @pytest.mark.skip()
    def test_analyze_predict_error(self, mq_logger):
        assert False

    @pytest.mark.skip()
    def test_analyze_predict_score_threshold(self, mq_logger):
        ts = np.exp([0.5, 0.6, 0.7])
        # for t in ts:
        #     check_predict_statistics()
        assert False

    @pytest.mark.skip()
    def test_TTA(self, mq_logger):
        assert False

    @pytest.mark.skip()
    def test_L_loss(self, mq_logger):
        assert False


if "__main__" == __name__:
    unittest.main()
