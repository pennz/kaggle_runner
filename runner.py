import pdb
import time

import parse
import pysnooper
from python_logging_rabbitmq import RabbitMQHandler

import utils


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
        pdb.set_trace()

    def _attach_data_collector(self, kernel):
        """
        Credits: https: // github.com/albertomr86/python-logging-rabbitmq

        !pip install python_logging_rabbitmq

        """

        logger = utils.get_logger()
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
        logger.addHandler(rabbit)

        cnt = 0
        while cnt < 50:
            time.sleep(1)
            logger.debug("hello test")
            cnt += 1

        # kernel.set_logger(self.kernel_name, handler=rabbit)
