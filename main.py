#!/usr/bin/env python3
import argparse

if "__main__" == __name__:
    parser = argparse.ArgumentParser(
        description="Main entry for computing worker.")
    parser.add_argument(
        "AMQPURL", type=str, help="AMQP URL to send log",
    )
    parser.add_argument(
        "integers",
        metavar="S",
        type=int,  # nargs='+',
        help="image size for inputing to network",
    )
    parser.add_argument(
        "integers",
        metavar="R",
        type=int,  # nargs='+',
        help="random seed",  # TODO we should just let it random internal
    )

    args = parser.parse_args()
