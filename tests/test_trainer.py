import os
import shutil
import subprocess
from unittest import TestCase

import pytest
from kaggle_runner.runners import trainer
from kaggle_runner import utils


@pytest.fixture(scope="module")
def runner_configs():
    return [
        {"size": 384, "network": "intercept", "AMQPURL": utils.AMQPURL()},
        {"size": 384, "network": "intercept-resnet", "AMQPURL": utils.AMQPURL()},
    ]

class TestTrainer(TestCase):
    @classmethod
    def setup_class(cls):
        cls.tmp_path = ".runners"
        print("setup_class called once for the class")

    @classmethod
    def teardown_class(cls):
        print("teardown_class called once for the class")

    def setup_method(self, method):
        if os.path.exists(self.tmp_path):
            shutil.rmtree(self.tmp_path)
            os.mkdir(self.tmp_path)
        print("setup_method called for every method")

    def teardown_method(self, method):
        # shutil.rmtree(self.tmp_path)  # for debug
        print("teardown_method called for every method")
    def test_forward(self):
        self.fail()

    def test_iterate(self):
        self.fail()

    def test_start(self):
        self.fail()
