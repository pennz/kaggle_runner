#!/usr/bin/env python3
from subprocess import call

AMQPURL = "amqp://guest:guest@127.0.0.1/"

with open("runner.sh", "w") as f:
    f.write(
        """#!/bin/bash
USER=$1
shift
REPO=$1
shift
BRANCH=$1
shift
PHASE=$1
shift
PARAMS=$@

( test -d ${REPO} || git clone --depth=1 \
https://github.com/${USER}/${REPO}.git ) && cd ${REPO} && \
([[ x$(git rev-parse --abbrev-ref HEAD) == x${BRANCH} ]] || \
git checkout -b ${BRANCH} --track origin/${BRANCH} ) && \
( [[ x"$PHASE" == x"dev" ]]  && python -m pytest -v) && python main.py $PARAMS
"""
    )
call(
    [
        "bash",
        "-x",
        "runner.sh",
        "pennz",
        "PneumothoraxSegmentation",
        "master",
        "dev",  # phase
        AMQPURL,
        "384",  # size 256+128
    ]
)
