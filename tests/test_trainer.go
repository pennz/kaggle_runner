import os
import shutil
import subprocess

import pytest

from kaggle_runner import trainer, utils


@pytest.fixture(scope="module")
def runner_configs():
    return [
        {"size": 384, "network": "intercept", "AMQPURL": utils.AMQPURL()},
        {"size": 384, "network": "intercept-resnet", "AMQPURL": utils.AMQPURL()},
    ]


class TestTrainer:
    coordinator = None
    tmp_path = "."

    @classmethod
    def setup_class(cls):
        cls.tmp_path = ".runners"
        cls.coordinator = coordinator.Coordinator(cls.tmp_path, "Test Runner")
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

    @pytest.mark.timeout(15)
    def test_push_runner_nb(self, runner_configs):
        path = self.coordinator.create_runner(runner_configs[1], 19999, False)
        # ret = self.coordinator.run_local(path)
        # assert ret.returncode == 0
        if os.getenv("CI") != "true":
            ret = self.coordinator.push(path)  # just push first
            assert ret.returncode == 0
