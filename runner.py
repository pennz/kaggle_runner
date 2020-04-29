import pdb

from python_logging_rabbitmq import RabbitMQHandler

import utils


class AMQPURL:
    # host = "termite.rmq.cloudamqp.com"  # (Load balanced)
    # passwd = "QrBHPPxbsd8IuIxKrCnX3-RGoLKaFhYI"
    # username = "drdsfaew"
    # Vhost = "drdsfaew"
    host = "127.0.0.1"  # (Load balanced)
    passwd = "guest"
    username = "guest"
    Vhost = "/"


class Runner:  # blade runner
    """Runner should run in docker container, and then a controller(or just mq
    to check the running status)
    """

    def __init__(self, kernel_code="TestKernel", AMQPURL=None, **kwargs):
        self.kernel_name = kernel_code
        self.AMQPURL = AMQPURL

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

        pdb.set_trace()
        logger.debug("test")
        logger.debug("more test")

        # kernel.set_logger(self.kernel_name, handler=rabbit)
