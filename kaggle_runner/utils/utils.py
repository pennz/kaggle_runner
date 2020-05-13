import logging

import parse


class AMQPURL:
    class __AMQPURL_DEV:
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
        host=__AMQPURL_DEV.host,
        passwd=__AMQPURL_DEV.passwd,
        Vhost=__AMQPURL_DEV.Vhost,
        username=__AMQPURL_DEV.username,
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
        return AMQPURL(host, passwd, Vhost, username)
    except TypeError as e:
        logger.debug(e)
