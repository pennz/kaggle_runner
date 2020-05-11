import json
import types
from socket import gethostname

import parse
import pysnooper
from python_logging_rabbitmq import RabbitMQHandler
from python_logging_rabbitmq.compat import text_type

from kaggle_runner import utils


def format(self, record):
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


def parse_AMQP(url_str):
    Vhost = None
    res = parse.parse("amqp://{}:{}@{}/{}", url_str)
    if res is None:
        res = parse.parse("amqp://{}:{}@{}/", url_str)
        if res is None:
            raise RuntimeError("AMQP URL error")
        else:
            Vhost = "/"
            username, passwd, host = res
    else:
        username, passwd, host, Vhost = res

    try:
        return utils.AMQPURL(host, passwd, Vhost, username)
    except TypeError as e:
        utils.logger.debug(e)


class Runner:  # blade runner
    """Runner should run in docker container, and then a controller(or just mq
    to check the running status)
    """

    def __init__(self, kernel_code="TestKernel", AMQPURL=None, **kwargs):
        self.kernel_name = kernel_code
        self.AMQPURL = parse_AMQP(AMQPURL)
        self.logger = None

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
        rabbit.formatter.format = types.MethodType(format, rabbit.formatter)
        logger.addHandler(rabbit)
        self.logger = logger
        # kernel.set_logger(self.kernel_name, handler=rabbit)
