from python_logging_rabbitmq import RabbitMQHandler


class Runner:  # blade runner
    """Runner should run in docker container, and then a controller(or just mq
    to check the running status)
    """

    def __init__(self, kernel_code="TestKernel"):
        self.kernel_name = kernel_code

    def _attach_data_collector(self, kernel):
        """
        Credits: https: // github.com/albertomr86/python-logging-rabbitmq

        !pip install python_logging_rabbitmq

        """

        rabbit = RabbitMQHandler(host="localhost", port=5672)
        # create kernel and run
        kernel.set_logger(self.kernel_name, handler=rabbit)
