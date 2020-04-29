import json
import os
import re
import shutil
from string import Template
from subprocess import call

import parse
import slug

import utils


class Coordinator:
    template_path = "runner_template/"
    """run in controller side, the runners run in dockers with GPUs"""

    def __init__(self, tmp_path, title_prefix):
        self.tmp_path = tmp_path
        self.runners = []
        self.title_prefix = title_prefix

    def push_all(self):
        for path in self.runners:
            self.push(path)

    @staticmethod
    def push(runner):
        "Push the code to server/kagger docker"
        utils.logger.debug(
            " ".join(["kaggle", "kernels", "push", "-p", runner]))
        return call(["kaggle", "kernels", "push", "-p", runner])

    def push_listen(self):
        self.push_all()
        self._get_result()

    def _get_result(self, timeout):
        """use the message queue, just use this right after push, listen for
        result, debug local first"""
        "use RE change source code to add the log collector"

    @staticmethod
    def _change_kernel_meta_info(folder, name):
        with open(os.path.join(folder, "kernel-metadata.json"), "r+") as jf:
            data = json.load(jf)
            slug_name = slug.slug(name)
            data["id"] = re.sub(r"/.*", "/" + slug_name, data["id"])
            data["title"] = slug_name
            jf.seek(0)
            json.dump(data, jf)
            jf.truncate()

    @staticmethod
    def _change_main_py(path, size, net, AMQPURL, seed):
        s = Template(
            """#!/usr/bin/env python3
from subprocess import call

with open("runner.sh", "w") as f:
    f.write(
        \"\"\"#!/bin/bash
USER=$1
shift
REPO=$1
shift
BRANCH=$1
shift
PHASE=$1
shift
PARAMS=$@

pip install slug parse # should move local codes out
pip install pysnooper python_logging_rabbitmq

( test -d ${REPO} || git clone --depth=1 \
https://github.com/${USER}/${REPO}.git ) && cd ${REPO} && \
([[ x$(git rev-parse --abbrev-ref HEAD) == x${BRANCH} ]] || \
git checkout -b ${BRANCH} --track origin/${BRANCH} ) && \
python main.py $PARAMS
\"\"\"
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
        "$AMQPURL",
        "$size",  # size 256+128
        "$seed",  # size 256+128
        "$network",
    ]
)
"""
        )

        d = dict(AMQPURL=AMQPURL.string(), size=size, network=net, seed=seed)
        ss = s.safe_substitute(d)

        with open(os.path.join(path, "main.py"), "w") as jf:
            jf.write(ss)

    def create_runner(self, config, seed="2020"):
        """
        config will be size and model right now
        """
        size = config["size"]
        net = config["network"]
        name = net.replace("_", "-") + "-" + str(size)
        AMQPURL = config["AMQPURL"]

        path = os.path.join(self.tmp_path, name)
        shutil.copytree(self.template_path, path)
        self._change_kernel_meta_info(path, self.title_prefix + " " + name)
        self._change_main_py(path, size, net, AMQPURL, seed)

        self.runners.append(path)

        return path
