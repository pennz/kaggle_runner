import logging
from abc import ABCMeta, abstractmethod

from kaggle_runner.kernels.KernelRunningState import KernelRunningState
from kaggle_runner.utils import kernel_utils

# Plot inline
# %matplotlib inline


class KernelGroup:
    """Kernel Group to try different combination of kernels hyperparameter

    Follow this example:
    https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

    Attributes:
        kernels (list): List of kernels whose results will be analyse together.

    """

    def __init__(self, kernels):
        """__init__.

        Args:
            kernels: List of kernels whose results will be analyse together.
        """
        self.kernels = kernels

class KaggleKernel(metaclass=ABCMeta):
    """Kernel for kaggle competitions/researchs

    Attributes:
        _stage
        model_metrics
        data_loader
        optimizer
        model
        train_X
        developing
        submit_run
        test_X
        model_loss
        num_epochs
        device
        result_analyzer
        logger
        dependency
        dev_X
        train_Y
        dev_Y

    """
    def __init__(self, logger=None):
        """__init__.

        Args:
            logger:
        """
        self.submit_run = False
        self.num_epochs = 8
        self.device = None
        self.optimizer = None
        self.data_loader = None
        self.developing = True
        self.model = None
        self.model_metrics = []
        self.model_loss = None

        self.train_X = None
        self.train_Y = None

        self.dev_X = None
        self.dev_Y = None

        self.test_X = None

        self.result_analyzer = None  # for analyze the result

        self._stage = KernelRunningState.INIT_DONE
        self.logger = logger
        self.dependency = []

    def _add_dependency(self, dep):
        """_add_dependency just install pip dependency now
        """
        self.dependency.append(dep)

    def install_dependency(self, dep):
        """install_dependency.

        Args:
            dep:
        """
        self._add_dependency(dep)

    def _add_logger_handler(self, handler):
        """_add_logger_handler.

        Args:
            handler:
        """
        self.logger.addHandler(handler)

    def set_logger(self, name, level=logging.DEBUG, handler=None):
        """set_logger.

        Args:
            name:
            level:
            handler:
        """
        FORMAT = "[%(levelname)s]%(asctime)s:%(name)s:%(message)s"
        logging.basicConfig(format=FORMAT)
        logger = logging.getLogger(name)
        logger.setLevel(level)

        if handler is not None:
            logger.addHandler(handler)
        self.logger = logger

    def set_random_seed(self):
        """set_random_seed.
        """
        pass

    def set_data_size(self):
        "might be useful when test different input datasize"
        pass

    def save_model(self):
        """save_model.
        """
        pass

    def load_model_weight(self):
        """load_model_weight.
        """
        pass

    @abstractmethod
    def build_and_set_model(self):
        """build_and_set_model.
        """
        pass

    def train_model(self):
        """train_model.
        """
        pass

    def set_model(self):
        """set_model.
        """
        pass

    def set_loss(self, loss_func):
        """set_loss.

        Args:
            loss_func:
        """
        pass

    def set_metrics(self, metrics):
        """
        set_metrics for model training

        :return: None
        """

    def set_result_analyzer(self):
        """set_result_analyzer.
        """
        pass

    def pre_prepare_data_hook(self):
        """pre_prepare_data_hook.
        """
        pass

    def after_prepare_data_hook(self):
        """after_prepare_data_hook.
        """
        pass

    def prepare_train_dev_data(self):
        """prepare_train_dev_data.
        """
        pass

    def prepare_test_data(self, data_config=None):
        """prepare_test_data.

        Args:
            data_config:
        """
        pass

    @abstractmethod
    def peek_data(self):
        """peek_data.
        """
        pass

    def predict_on_test(self):
        """predict_on_test.
        """
        pass

    def dump_state(self, exec_flag=False):
        """dump_state.

        Args:
            exec_flag:
        """
        self.logger.debug("state %s" % self._stage)

        if exec_flag:
            self.logger.debug("dumping state to file for %s" % self._stage)
            # dump_obj(self, 'run_state.pkl', force=True)  # too large
            kernel_utils.dump_obj(self, "run_state_%s.pkl" %
                                  self._stage, force=True)

    def run(
        self,
        start_stage=None,
        end_stage=KernelRunningState.SAVE_SUBMISSION_DONE,
        dump_flag=False,
    ):
        """

        :param start_stage: if set, will overwrite the stage
        :param end_stage:
        :param dump_flag:
        :return:
        """
        self.continue_run(
            start_stage=start_stage, end_stage=end_stage, dump_flag=dump_flag
        )

    def continue_run(
        self,
        start_stage=None,
        end_stage=KernelRunningState.SAVE_SUBMISSION_DONE,
        dump_flag=False,
    ):
        """continue_run.

        Args:
            start_stage:
            end_stage:
            dump_flag:
        """
        self.set_random_seed()
        self.logger.debug(
            "%s -> %s", start_stage, end_stage,
        )

        if start_stage is not None:
            assert start_stage.value < end_stage.value
            self._stage = start_stage

        if self._stage.value < KernelRunningState.PREPARE_DATA_DONE.value:
            self.pre_prepare_data_hook()
            self.prepare_train_dev_data()
            self.after_prepare_data_hook()

            self._stage = KernelRunningState.PREPARE_DATA_DONE
            self.dump_state(exec_flag=dump_flag)

            if self._stage.value >= end_stage.value:
                return

        if self._stage.value < KernelRunningState.TRAINING_DONE.value:
            self.pre_train()
            self.build_and_set_model()
            self.train_model()
            self.after_train()

            self.save_model()  # during training, it will also save model

            self._stage = KernelRunningState.TRAINING_DONE
            self.dump_state(exec_flag=dump_flag)

            if self._stage.value >= end_stage.value:
                return

        if self._stage.value < KernelRunningState.EVL_DEV_DONE.value:
            self.set_result_analyzer()

            self._stage = KernelRunningState.EVL_DEV_DONE
            self.dump_state(exec_flag=dump_flag)

            if self._stage.value >= end_stage.value:
                return

        if self._stage.value < KernelRunningState.SAVE_SUBMISSION_DONE.value:
            self.pre_test()
            self.prepare_test_data()
            self.predict_on_test()
            self.after_test()

            self.pre_submit()
            self.submit()
            self.after_submit()

            self._stage = KernelRunningState.SAVE_SUBMISSION_DONE
            self.dump_state(exec_flag=dump_flag)

            if self._stage.value >= end_stage.value:
                return

    @classmethod
    def _load_state(cls, stage=None, file_name="run_state.pkl", logger=None):
        """

        :param file_name:
        :return: the kernel object, need to continue
        """

        if stage is not None:
            file_name = f"run_state_{stage}.pkl"

        if logger is not None:
            logger.debug(f"restore from {file_name}")
        self = kernel_utils.get_obj_or_dump(filename=file_name)
        assert self is not None
        self.logger = logger

        return self

    def load_state_data_only(self, file_name="run_state.pkl"):
        """load_state_data_only.

        Args:
            file_name:
        """
        pass

    @classmethod
    def load_state_continue_run(cls, file_name="run_state.pkl", logger=None):
        """

        :param file_name:
        :return: the kernel object, need to continue
        """
        self = cls._load_state(file_name=file_name, logger=logger)
        self.continue_run()

    def pre_train(self):
        """pre_train.
        """
        pass

    def after_train(self):
        """after_train.
        """
        pass

    def pre_submit(self):
        """pre_submit.
        """
        pass

    def submit(self):
        """submit.
        """
        pass

    def after_submit(self):
        "after_submit should report to our logger, for next step analyze"
        pass

    def pre_test(self):
        """pre_test.
        """
        pass

    def after_test(self):
        """after_test.
        """
        pass

    def save_result(self):
        """save_result.
        """
        pass

    def plot_train_result(self):
        """plot_train_result.
        """
        pass

    def plot_test_result(self):
        """plot_test_result.
        """
        pass

    def analyze_data(self):
        """analyze_data.
        """
        pass

    @abstractmethod
    def check_predict_details(self):
        """check_predict_details.
        """
        pass

from kaggle_runner import logger

class KaggleKernelOnlyPredict(KaggleKernel):
    """KaggleKernelOnlyPredict.
    """


    def __init__(self, model_path):
        """__init__.

        Args:
            model_path:
        """
        super(KaggleKernelOnlyPredict, self).__init__(logger=logger)
        self.only_predict = True

    @abstractmethod
    def build_and_set_model(self):
        """load pretrained one"""
        pass

    def prepare_train_dev_data(self):
        """prepare_train_dev_data.
        """
        pass

    @abstractmethod
    def prepare_test_data(self, data_config=None):
        """prepare_test_data.

        Args:
            data_config:
        """
        pass

    @abstractmethod
    def check_predict_details(self):
        """check_predict_details.
        """
        pass

    def peek_data(self):
        """peek_data.
        """
        pass


def test_init_only_predict():
    """test_init_only_predict.
    """
    k = KaggleKernelOnlyPredict()
    assert k is not None
    assert k.model is not None
