import json
import types
from socket import gethostname

from kaggle_runner import kernels
from kaggle_runner.runners.trainer import Trainer
from kaggle_runner.utils import utils
from kaggle_runner.utils.utils import parse_AMQP
from python_logging_rabbitmq import RabbitMQHandler
from python_logging_rabbitmq.compat import text_type


def log_format(self, record):
    data = record.__dict__.copy()

    if record.args:
        msg = record.msg % record.args
    else:
        msg = record.msg

    data.update(
        host=gethostname(), msg=msg, args=tuple(text_type(arg) for arg in record.args)
    )

    if "exc_info" in data and data["exc_info"]:
        data["exc_info"] = self.formatException(data["exc_info"])

    if self.include:
        data = {f: data[f] for f in self.include}
    elif self.exclude:
        for f in self.exclude:
            if f in data:
                del data[f]

    return json.dumps(data)


class TrainerConfig:
    "TrainerConfig control the process of the training -> runner call trainer -> then kernel"
    " for complicate logic, how can it be implemented? just patch, or just another layer of abastraction to "
    " handle training process"

    "No just simple logic, the aim is calculating fast first"
    pass


class TrainerWithStatus(Trainer):
    def __init__(self, model, data_folder, df_path, config=None):
        assert isinstance(model, kernels.KaggleKernel)
        super(Trainer, self).__init__(model, data_folder, df_path)
        self._handle_config(config)

    def _handle_config(self, config):
        pass


class Runner:  # blade runner
    """Runner should run in docker container, and then a controller(or just mq) will check the running status
    """

    def __init__(self, kernel_code="TestKernel", AMQPURL=None, **kwargs):
        self.kernel_name = kernel_code
        self.AMQPURL = parse_AMQP(AMQPURL)
        self.logger = None
        self.trainer = None

    def _attach_data_collector(self, kernel):
        """
        Credits: https: // github.com/albertomr86/python-logging-rabbitmq

        !pip install python_logging_rabbitmq

        """

        logger = utils.get_logger(self.kernel_name + "_runner")
        rabbit = RabbitMQHandler(
            host=self.AMQPURL.host,
            port=5672,
            username=self.AMQPURL.username,
            password=self.AMQPURL.passwd,
            exchange="logs_topic",
            connection_params={"virtual_host": self.AMQPURL.Vhost},
            declare_exchange=True,
        )
        # rabbit.connection_params["virtual_host"] = self.AMQPURL.Vhost create
        # kernel and run
        rabbit.formatter.format = types.MethodType(
            log_format, rabbit.formatter)
        logger.addHandler(rabbit)
        self.logger = logger
        # kernel.set_logger(self.kernel_name, handler=rabbit)
