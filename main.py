#!/usr/bin/env python3
import argparse

from kaggle_runner.runners.runner import Runner

if "__main__" == __name__:
    parser = argparse.ArgumentParser(
        description="Main entry for computing worker.")
    parser.add_argument(
        "AMQPURL", metavar="A", type=str, help="AMQP URL to send log",
    )
    parser.add_argument(
        "size",
        metavar="S",
        type=int,  # nargs='+',
        help="image size for inputing to network",
    )
    parser.add_argument(
        "seed",
        metavar="E",
        type=int,  # nargs='+',
        help="random seed",  # TODO we should just let it random internal
    )
    parser.add_argument(
        "network", metavar="N", type=str, help="network type",  # nargs='+',
    )

    args = parser.parse_args()
    r = Runner(args.network, args.AMQPURL, size=args.size, seed=args.seed)
    assert r.AMQPURL is not None
    # TODO should push the runner and get message back
    r._attach_data_collector(None)
    r.logger.debug("Runner created")
