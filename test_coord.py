import importlib
import os
import time
import shutil
import sys
import unittest

import numpy as np
import torch
from IPython.core.debugger import set_trace
from torch.autograd import Variable

import kernel

# import modelTester
import pytorchKernel

# from kernel import Kernel
import utils
from kernel import KernelRunningState
import coordinator
import pytest


class CoordinatorTest():
    @classmethod
    def setup_class(cls):
        cls.tmp_path = ".runners"
        cls.coordinator = coordinator.Coordinator()
        print("setup_class called once for the class")

    @classmethod
    def teardown_class(cls):
        print("teardown_class called once for the class")


    def setup_method(self, method):
        shutil.rmtree(self.tmp_path)
        os.mkdir(self.tmp_path)
        print("setup_method called for every method")

    def teardown_method(self, method):
        shutil.rmtree(self.tmp_path)
        print("teardown_method called for every method")

    @pytest.mark.timeout(5)
    def test_get_mq_back(self):
        print("one")
        self.coordinator.create_runner()
        self.coordinator.push()
        # just use a timeout, not within then return error
        time.sleep(10)
        self.coordinator._wait_feedback(timeout=100)

    def test_two(self):
        print("two")
        assert False
        print("two after")

    def test_three(self):
        print("three")
        assert True
        print("three after")

    def test_create_runners(self, runner_configs):
        """Should just use unit test setup and teardown
        """
        for c in runner_configs:
            r = self.coordinator.create_runner(c)
        assert r.AMQPURL is not None

