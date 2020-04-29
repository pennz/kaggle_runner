import coordinator
import pytest
import shutil
import os
from subprocess import call
import utils


@pytest.fixture(scope="module")
def runner_configs():
    return [
        {"size": 384, "network": "intercept", "AMQPURL": utils.AMQPURL()},
        {"size": 384, "network": "intercept-resnet", "AMQPURL": utils.AMQPURL()},
    ]


class TestCoordinator:
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

    @pytest.mark.timeout(5)
    def test_push_runner(self, runner_configs):
        path = self.coordinator.create_runner(runner_configs[1], 19999)
        ret = self.coordinator.push(path)
        assert ret == 0
        # just use a timeout, not within then return error

    @pytest.mark.timeout(10)
    def test_get_mq_back(self, runner_configs):
        path = self.coordinator.create_runner(runner_configs[1], 20202)
        ret = self.coordinator.push(path)
        assert ret == 0
        # just use a timeout, not within then return error
        self.coordinator._get_result(timeout=100)

    @pytest.mark.skip("runner runs in computation server, no need test local")
    def test_create_runners(self, runner_configs):
        """Should just use unit test setup and teardown
        """
        for c in runner_configs:
            r = self.coordinator.create_runner(c)
        assert r.AMQPURL is not None


class TestMain:
    def test_call(self):
        call_params = [
            "python",
            "main.py",
            "amqp://guest:guest@127.0.0.1/",
            "384",  # size 256+128
            "123",
            "intercept-resnet",
        ]
        utils.logger.debug(" ".join(call_params))
        ret = call(call_params)
        assert ret == 0
