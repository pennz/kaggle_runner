import logging


class AMQPURL:
    class AMQPURL_DEV:
        host = "termite.rmq.cloudamqp.com"  # (Load balanced)
        passwd = "QrBHPPxbsd8IuIxKrCnX3-RGoLKaFhYI"  # oh~ just give my password out~
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


def get_logger(name="utils", level=logging.DEBUG):
    "get_logger just return basic logger, no AMQP included"
    FORMAT = "[%(levelname)s]%(asctime)s:%(name)s:%(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
