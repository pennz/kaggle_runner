from python_logging_rabbitmq import RabbitMQHandler

import utils


class Runner:  # blade runner
    """Runner should run in docker container, and then a controller(or just mq
    to check the running status)
    """

    def __init__(self, kernel_code="TestKernel", AMQPURL="", **kwargs):
        self.kernel_name = kernel_code
        self.AMQPURL = AMQPURL

    def _attach_data_collector(self, kernel):
        """
        Credits: https: // github.com/albertomr86/python-logging-rabbitmq

        !pip install python_logging_rabbitmq

        """

        logger = utils.get_logger()
        rabbit = RabbitMQHandler(host="localhost", port=5672)
        # create kernel and run
        logger.addHandler(rabbit)
        logger.debug("test")
        # kernel.set_logger(self.kernel_name, handler=rabbit)
