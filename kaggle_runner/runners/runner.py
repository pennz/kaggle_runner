from kaggle_runner.utils import utils
from kaggle_runner.utils.utils import parse_AMQP


class Runner:  # blade runner
    """Runner should run in docker container, and then a controller(or just mq) will check the running status
    """
    def __init__(self, kernel_code="TestKernel", AMQPURL=None, **kwargs):
        self.kernel_name = kernel_code
        self.AMQPURL = parse_AMQP(AMQPURL)
        self.logger = None
        self.trainer = None

    def _attach_data_collector(self, kernel=""):
        """
        Credits: https: // github.com/albertomr86/python-logging-rabbitmq

        !pip install python_logging_rabbitmq

        """
        logger = utils.get_logger(self.kernel_name + "_runner")
        self.logger = utils.attach_data_collector(logger, self.AMQPURL)
