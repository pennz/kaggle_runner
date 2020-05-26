import json
import logging
import types
from socket import gethostname

import parse
from python_logging_rabbitmq import RabbitMQHandler
from python_logging_rabbitmq.compat import text_type


class AMQPURL:
    class __AMQPURL_DEV:
        #host = "termite.rmq.cloudamqp.com"  # (Load balanced)
        #passwd = "QrBHPPxbsd8IuIxKrCnX3-RGoLKaFhYI"  # oh~ just give my password out~
        #username = "drdsfaew"
        #Vhost = "drdsfaew"
        host = "pengyuzhou.com"
        passwd = "9b83ca70cf4cda89524d2283a4d675f6"
        username = "kaggle"
        Vhost = "/"

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

def attach_data_collector(logger, AMQPURL=AMQPURL()):
    """
    Credits: https: // github.com/albertomr86/python-logging-rabbitmq

    !pip install python_logging_rabbitmq

    """

    rabbit = RabbitMQHandler(
        host=AMQPURL.host,
        port=5672,
        username=AMQPURL.username,
        password=AMQPURL.passwd,
        exchange="logs_topic",
        connection_params={"virtual_host": AMQPURL.Vhost},
        declare_exchange=True,
    )
    # rabbit.connection_params["virtual_host"] = self.AMQPURL.Vhost create
    # kernel and run
    rabbit.formatter.format = types.MethodType(
        log_format, rabbit.formatter)
    logger.addHandler(rabbit)

    return logger

logger = attach_data_collector(get_logger())

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
