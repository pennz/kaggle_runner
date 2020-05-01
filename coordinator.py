import json
import os
import re
import shutil
from string import Template
from subprocess import call

import pysnooper
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
    def _change_kernel_meta_info(folder, name, script):
        with open(os.path.join(folder, "kernel-metadata.json"), "r+") as jf:
            data = json.load(jf)
            if not script:
                name = name + " nb"
            slug_name = slug.slug(name)
            data["id"] = re.sub(r"/.*", "/" + slug_name, data["id"])
            data["title"] = slug_name
            if not script:
                data["kernel_type"] = "notebook"
                data["code_file"] = "main.ipynb"
            else:
                data["kernel_type"] = "script"
                data["code_file"] = "main.py"
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

pip install pydicom
pip install parse # should move local codes out
pip install pytest-logger pysnooper python_logging_rabbitmq  # for debugging

(test -d ${REPO} || git clone --single-branch --branch ${BRANCH} --depth=1 \
https://github.com/${USER}/${REPO}.git ${REPO} && pushd ${REPO} && \
find . -maxdepth 1 -name ".??*" -o -name "??*" | xargs -I{} cp -r {} $OLDPWD && popd) && \
{ if [ x"${PHASE}" != x"dev" ]; then python main.py $PARAMS; fi }
\"\"\"
    )
call(
    [
        "bash",
        "runner.sh",
        "pennz",
        "PneumothoraxSegmentation",
        "dev",
        "dev",  # phase
        "$AMQPURL",
        "$size",  # size 256+128
        "$seed",  # size 256+128
        "$network",
    ]
)

# %run -m pytest -v
"""
        )

        d = dict(AMQPURL=AMQPURL.string(), size=size, network=net, seed=seed)
        ss = s.safe_substitute(d)

        with open(os.path.join(path, "main.py"), "w") as jf:
            jf.write(ss)

    @pysnooper.snoop()
    def create_runner(self, config, seed="2020", script=True):
        """
        config will be size and model right now
        """
        size = config["size"]
        net = config["network"]
        name = net.replace("_", "-") + "-" + str(size)
        AMQPURL = config["AMQPURL"]

        path = os.path.join(self.tmp_path, name)
        shutil.copytree(self.template_path, path)
        self._change_kernel_meta_info(
            path, self.title_prefix + " " + name, script)
        self._change_main_py(path, size, net, AMQPURL, seed)
        if not script:
            call(("jupytext --to notebook " + os.path.join(path, "main.py")).split())

        self.runners.append(path)

        return path
