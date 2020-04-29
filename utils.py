import logging


class AMQPURL:
    class AMQPURL_DEV:
        host = "termite.rmq.cloudamqp.com"  # (Load balanced)
        passwd = "QrBHPPxbsd8IuIxKrCnX3-RGoLKaFhYI"
        username = "drdsfaew"
        Vhost = "drdsfaew"
        # host = "127.0.0.1"  # (Load balanced)
        # passwd = "guest"
        # username = "guest"
        # Vhost = "/"

    def __init__(
        self,
        host=AMQPURL_DEV.host,
        passwd=AMQPURL_DEV.passwd,
        Vhost=AMQPURL_DEV.Vhost,
        username=AMQPURL_DEV.username,
    ):
        self.host = host
        self.passwd = passwd
        self.Vhost = Vhost
        self.username = username

    def string(self):
        Vhost = self.Vhost
        if self.Vhost == "/":
            Vhost = ""
        return f"amqp://{self.username}:{self.passwd}@{self.host}/{Vhost}"


def get_logger():
    FORMAT = "[%(levelname)s]%(asctime)s:%(name)s:%(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    return logger


logger = get_logger()
