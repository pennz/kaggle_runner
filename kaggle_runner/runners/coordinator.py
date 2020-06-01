import json
import os
import re
import shutil
import subprocess
import sys
from string import Template

import slug

from kaggle_runner.utils import AMQPURL, logger

rvs_str = r"""#!/bin/bash -x
export PS4='Line ${LINENO}: ' # for debug
NC=ncat

# https://stackoverflow.com/questions/57877451/retrieving-output-and-exit-code-of-a-coprocess
# coproc { sleep 30 && echo "Output" && exit 3; }
# Saving the coprocess's PID for later, as COPROC_PID apparently unsets when its finished
# COPROC_PID_backup=$COPROC_PID
#
# Retrieving the coprocess's output
# output=$(cat <&$COPROC)
#
# Retrieving the coprocess's exit code
# wait $COPROC_PID_backup
#
# Echoing out the results
# echo $?
# echo $output

echo BASH NOW: $BASHPID

PID_FILE_PATH=/tmp/nc.pid
EXIT_FILE_PATH=/tmp/rvs_exit.$BASHPID.pid

test -f $EXIT_FILE_PATH && rm $EXIT_FILE_PATH

SERVER=$1
shift
PORT=$1
shift

ORIG_PORT=23454
CHECK_PORT=$((ORIG_PORT + 1))

check_exit_status() {
  [ -f /tmp/rvs_return ] && return 0

  if [ -f $EXIT_FILE_PATH ] && [ x"$(cat $EXIT_FILE_PATH)" = x0 ]; then
    return 0
  fi

  return 1 # not ok
}


connect_setup() {
  connect_again_flag=1

  sleep_time=5

  while [ ${connect_again_flag} -eq 1 ]; do
    check_exit_status && return 0

    $NC -w ${1}s -i 1800s $SERVER $PORT -c "echo $(date) started connection; echo $HOSTNAME; python -c 'import pty; pty.spawn([\"/bin/bash\", \"-li\"])'"

    RSRET=$?
    echo $RSRET > $EXIT_FILE_PATH
    >&2 echo "$NC return with code $RSRET"

    if [ x"$RSRET" = x"0" ]; then
      [ -f /tmp/rvs_exit ] && return 0

      return 255 # just do not return
    fi
    [ $RSRET -eq 0 ] && connect_again_flag=0
    [ $RSRET -eq 1 ] && sleep ${sleep_time} && sleep_time=$((sleep_time + sleep_time))
  done
  # exit, will cause rvs script exit, beside, RSRET not 0, mean connection loss
  # thing
  RSRET=1  # just never exit
  echo $RSRET > $EXIT_FILE_PATH && return $RSRET
}

connect_again() {
  # pkill -f "nc.*$PORT"  # no need now, our listen server can accept multiple
  # connection now
  connect_setup $1
}

WAIT_LIMIT=2048
INIT_WAIT=8
port_connect_status=0
wait_time=$INIT_WAIT

floatToInt() {
  parsed=$(printf "%.0f" "$@")
  [ ! $? -eq 0 ] && parsed=0
  echo $parsed
} 2> /dev/null

while true; do
  check_exit_status && exit 0
  # if find that server cannot be connected, we try to restart our reverse connect again
  nc_time=$($(which time) -f "%e" $NC -zw $wait_time $SERVER $CHECK_PORT 2>&1 > /dev/null)
  nc_ret=$?
  nc_time=$(echo $nc_time | awk '{print $NF}')
  nc_time=$(floatToInt $nc_time)

  if [ ${nc_ret} -eq 0 ]; then
    # recover connection, need to connect_again too. For 1st time, will try to connect
    # no connection last time, have connction now

    if [ $port_connect_status -eq 0 ]; then
      echo "recover connection, reset wait_time and try to reconnect"
      wait_time=$INIT_WAIT
      # previous connection is lost, we wait for longer to setup connection
      check_exit_status || wait_time=15
      connect_again $wait_time &
    else
      wait_time=$((wait_time + wait_time)) # double wait, network fine

      if [ $wait_time -gt ${WAIT_LIMIT} ]; then wait_time=${WAIT_LIMIT}; fi
    fi
    port_connect_status=1
  else
    if [ $port_connect_status -eq 1 ]; then
      echo "found connection loss, reset wait_time and try to reconnect"
      wait_time=$INIT_WAIT
      check_exit_status || wait_time=15 # previous connection is lost
      connect_again $wait_time &
    else # no connection all the time? we still try to connect...
      wait_time=$((wait_time + wait_time))

      if [ $wait_time -gt ${WAIT_LIMIT} ]; then wait_time=${WAIT_LIMIT}; fi
      connect_again $wait_time &
    fi
    port_connect_status=0
  fi
  sleep $((wait_time - nc_time)) # check every XX seconds
  echo $hostname $HOSTNAME
done
wait  # wait for any background

# https://medium.com/@6c2e6e2e/spawning-interactive-reverse-shells-with-tty-a7e50c44940e
# In reverse shell
# $ python -c 'import pty; pty.spawn("/bin/bash")'
# Ctrl-Z
#
# In Attacker console
# $ stty raw -echo
# $ fg
#
# In reverse shell
# $ reset
# $ export SHELL=bash
# $ export TERM=xterm-256color
# $ stty rows <num> columns <cols>
"""

rvs_pty_config_str = r"""#!/bin/bash
[ -d ~/.fzf ] || {
git clone --depth=1 https://github.com/pennz/dotfiles
rsync -r dotfiles/.* ~
pushd ~
git submodule update --init
.fzf/install --all
curl -fLo ~/.config/nvim/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
curl -fLo ~/.vim/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
# vim -u ~/.vimrc_back "+call plug#begin()" +PlugInstall +qa &
# ( sleep 60; nvim -Vnvim_log -u ~/.vimrc_back "+call plug#begin()" +PlugInstall +checkhealth +qa )&
ln -s .shrc_customised.macos .shrc_customised
echo "alias gdrive='gdrive  --service-account a.json'" >> ~/.bash_aliases
echo "unalias vim" >> ~/.bash_aliases
echo "alias vim='nvim -u ~/.vimrc_back'" >> ~/.bash_aliases
popd

cat >> ~/.profile << EOF
reset
export SHELL=/bin/bash
export TERM=screen-256color
stty intr ^\c susp ^\x eof ^\f echo opost
stty rows 40 columns 120
# https://unix.stackexchange.com/questions/343088/what-is-the-equivalent-of-stty-echo-for-zsh
# unsetopt ZLE # for zsh
# for ourside stty raw isig -echo icrnl time 3 echoprt opost eof ^\p

color_my_prompt () {
    local __user_and_host="\[\033[01;32m\]\u@\h"
    local __cur_location="\[\033[01;34m\]\w"
    local __git_branch_color="\[\033[31m\]"
    # local __git_branch="\`ruby -e \"print (%x{git branch 2> /dev/null}.grep(/^\*/).first || '').gsub(/^\* (.+)$/, '(\1) ')\"\`"
    local __git_branch='`git branch 2> /dev/null | grep -e ^* | ${SED:-sed} -E  s/^\\\\\*\ \(.+\)$/\(\\\\\1\)\ /`'
    local __prompt_tail="\[\033[35m\]$"
    local __last_color="\[\033[00m\]"
    export PS1="$__user_and_host $__cur_location $__git_branch_color$__git_branch$__prompt_tail$__last_color "
}

ENV=/root/.bashrc
PYTHONWARNINGS=ignore:::pip._internal.cli.base_command
MPLBACKEND=module://ipykernel.pylab.backend_inline

PS4="$HOSTNAME: "'${LINENO}: '
_=/usr/bin/env
PWD=/kaggle/working
cd $PWD
OLDPWD=/root

# color_my_prompt
locale-gen
echo "#" $(grep 'cpu ' /proc/stat >/dev/null;sleep 0.1;grep 'cpu ' /proc/stat | awk -v RS="" '{print "CPU: "($13-$2+$15-$4)*100/($13-$2+$15-$4+$16-$5)"%"}') "Mem: "$(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{print 100-100*a/t"%"}' /proc/meminfo) "Uptime: "$(uptime | awk '{print $1 " " $2 " " $3}')
echo "#" TPU_NAME=$TPU_NAME
nvidia-smi
conda activate base
EOF
}
"""

gdrive_str = r"""#!/bin/bash
wget https://github.com/gdrive-org/gdrive/releases/download/2.1.0/gdrive-linux-x64
chmod +x gdrive-linux-x64
cp gdrive-linux-x64 /bin/gdrive

mkdir ~/.gdrive

# auth file
cat > ~/.gdrive/a.json << EOF
CONTENT_CREDENTIAL
EOF

gdrive --service-account a.json list  # just test

SRC_WORK_FOLDER=/kaggle/input
[ -d ${SRC_WORK_FOLDER} ] || {
    mkdir -p ${SRC_WORK_FOLDER}
    cd ${SRC_WORK_FOLDER}
    gdrive --service-account a.json download -r 1CHDWIN0M6PD4SQyplbWefBCzNzdPVd-m
    tar xf siim-train-test.tar.gz -C /kaggle/input
}
# cat > tgz_files.sh << EOF
# #!/bin/bash
# tgzfile () {
#   tar cf - $1 -P | pv -s $(du -sb $1 | awk '{print $1}') | gzip > /home/$1.tar.gz
# }
# cd /kaggle/input
# find . -maxdepth 1 -type d -name "??*" | while read -r line; do
#     echo $line
#     tgzfile $line
# done
# EOF
"""

runner_src = r"""#!/bin/bash -x
export PS4='Line ${LINENO}: ' # for debug
NC=ncat

USER=$1
shift
REPO=$1
shift
BRANCH=$1
shift
PHASE=$1
shift
ENABLE_RVS=$1
shift

SERVER=$1
shift
PORT=$1
shift

ORIG_PORT=23454

CHECK_PORT=$((ORIG_PORT + 1))
conda install -y -c eumetsat expect & # https://askubuntu.com/questions/1047900/unbuffer-stopped-working-months-ago
apt update && apt install -y netcat nmap screen time locales >/dev/null 2>&1
apt install -y mosh fish tig ctags htop tree pv tmux psmisc neovim expect >/dev/null 2>&1 &

conda init bash
cat >> ~/.bashrc << EOF
conda activate base # as my dotfiles will fiddle with conda
export SERVER=$SERVER
export CHECK_PORT=$CHECK_PORT
EOF

source rpt # rvs IDE env setup
export SERVER=$SERVER
export CHECK_PORT=$CHECK_PORT

wait_ncat() {
    wait_for_ncat=$1

    while [ $wait_for_ncat -gt 0 ]; do
        wait_for_ncat=$((wait_for_ncat - 1))
        which ncat >/dev/null && return 0
    done
}
wait_ncat 60

which $NC >/dev/null || NC=nc
export NC

pip install ripdb pydicom parse pytest-logger python_logging_rabbitmq coverage &
# python3 -m pip install pyvim neovim msgpack==1.0.0 &
# python -m pip install pyvim neovim msgpack==1.0.0 & # for vim

SRC_WORK_FOLDER=/kaggle/working
[ -d ${SRC_WORK_FOLDER} ] || mkdir -p ${SRC_WORK_FOLDER}

cd ${SRC_WORK_FOLDER}

if [ -d ${REPO} ]; then rm -rf ${REPO}; fi

# get code
{
    mvdir() {
        [[ "$2"/"$1" -ef "${PWD}" ]] || {
            rm -rf "$2"/"$1" &&
                mkdir "$2"/"$1"
        }

        bash -c "mv ""$1""/*"" $2""/""$1"
    }
    export -f mvdir

    git clone --single-branch --branch ${BRANCH} --depth=1 \
        https://github.com/${USER}/${REPO}.git ${REPO} && pushd ${REPO} &&
        git submodule update --init --recursive
    find . -maxdepth 1 -name ".??*" -o -name "??*" -type f | xargs -I{} mv {} $OLDPWD
    find . -maxdepth 1 -name ".??*" -o -name "??*" -type d | xargs -I{} bash -x -c "mvdir {}  $OLDPWD"
    popd
    pip install -e . &
    make install_dep >/dev/null
}

USE_AMQP=true
export USE_AMQP

conda init bash
source ~/.bashrc
conda activate base

if [ x"${PHASE}" = x"dev" ]; then
    export PS4='[Remote]: Line ${LINENO}: '
    (
        echo "MOSHing"
        make mosh
    ) &

    if [ "x${ENABLE_RVS}" = x1 ]; then
        make toxic | if [ $USE_AMQP -eq true ]; then cat -; else $NC --send-only -w 120s -i $((60 * 5))s $SERVER $CHECK_PORT; fi &

        if [ -z $(pgrep -f 'jupyter-notebook') ]; then
            bash ./rvs.sh $SERVER $PORT 2>&1 &
        else
            screen -d -m bash -c "{ echo [REMOTE]: rvs log below.; bash -x ./rvs.sh $SERVER $PORT 2>&1; } | $NC --send-only --no-shutdown -w 120s -i $((3600 * 2))s $SERVER $CHECK_PORT"
        fi
    fi &
    wait # not exit, when dev
fi

if [ x"${PHASE}" = x"data" ]; then
    bash ./rvs.sh $SERVER $PORT >/dev/null & make m & # just keep one rvs incase
    make dataset
    wait
fi

if [ x"${PHASE}" = x"run" ]; then
    #pip install kaggle_runner
    bash ./rvs.sh $SERVER $PORT >/dev/null & make m & # just keep one rvs incase
    make toxic | if [ $USE_AMQP -eq true ]; then cat -; else $NC --send-only -w 120s -i $((60 * 5))s $SERVER $CHECK_PORT; fi
    # basically the reverse of the calling path
    pkill make & pkill -f "mosh" & pkill sleep & pkill -f "rvs.sh" & pkill ncat &
    # python main.py "$@"
fi
"""


class Coordinator:
    template_path = "kaggle_runner/runner_template/"  # TODO just put it in the code
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
        comm = f"PATH=~/.local/bin:$PATH kaggle kernels push -p {runner}"
        logger.debug(comm)

        return subprocess.run(comm, shell=True)

    def push_listen(self):
        self.push_all()
        self._get_result(timeout=60)

    def _get_result(self, timeout):
        """use the message queue, just use this right after push, listen for
        result, debug local first

        use jq change source code to add the log collector"
        """

    @staticmethod
    def _change_kernel_meta_info(folder, name, script, gpu=False):
        with open(os.path.join(folder, "kernel-metadata.json"), "r+") as jf:
            data = json.load(jf)

            if name is not None:
                if not script:
                    name = name + " nb"
                slug_name = slug.slug(name)
                data["id"] = re.sub(r"/.*", "/" + slug_name, data["id"])
                data["title"] = slug_name
            data["enable_gpu"] = "true" if gpu else "false"

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
    def _change_main_py(path, size, net, AMQPURL, seed, port, gdrive_enable=False, phase='dev'):
        s = Template(
            """#!/usr/bin/env python3
from importlib import reload
import os
import selectors
import subprocess
import sys

subprocess.run('git clone https://github.com/pennz/kaggle_runner; rsync -r kaggle_runner/.* .; rsync -r kaggle_runner/* .; python -m pip install -e .', shell=True, check=True)
import kaggle_runner
reload(kaggle_runner)
from kaggle_runner import logger
logger.debug("Logger loaded")

with open("runner.sh", "w") as f:
    f.write(
r\"\"\"${runner_src}\"\"\"
    )
with open("rvs.sh", "w") as f:
    f.write(
r\"\"\"${rvs_str}\"\"\"
    )
with open("rpt", "w") as f:
    f.write(
r\"\"\"${rvs_pty_config_str}\"\"\"
    )
with open("gdrive_setup", "w") as f:
    f.write(
r\"\"\"${gdrive_str}\"\"\"
    )

server = "${server}"
os.environ['SERVER'] = server

entry_str = r\"\"\"#!/bin/bash
PS4='Line ${LINENO}: ' bash -x runner.sh pennz kaggle_runner master "$phase" 1 \"\"\"+ server +\"\"\" "$port" "$AMQPURL" "$size" "$seed" "$network" | tee runner_log
\"\"\"
if ${gdrive_enable}:
    entry_str += r\"\"\"PS4='Line ${LINENO}: ' bash -x gdrive_setup >>loggdrive &\"\"\"

with open("entry.sh", "w") as f:
    f.write(entry_str)


p = subprocess.Popen(
'bash -x entry.sh',
stdout=subprocess.PIPE, stderr=subprocess.PIPE,
shell=True)

sel = selectors.DefaultSelector()
sel.register(p.stdout, selectors.EVENT_READ)
sel.register(p.stderr, selectors.EVENT_READ)

while True:
   for key, _ in sel.select():
       data = key.fileobj.read1(1024).decode()
       if not data:
           exit()
       data = data.strip()
       if data == "":
           continue
       if key.fileobj is p.stdout:
           logger.debug(data)
           print(data, end="")
       else:
           logger.error(data)
           print(data, end="", file=sys.stderr)

# URL:
# https://stackoverflow.com/questions/31833897/python-read-from-subprocess-stdout-and-stderr-separately-while-pr
# eserving-order
# Title: Python read from subprocess stdout and stderr separately while preserving order - Stack Overflow
"""
        )

        try:
            gdpass = subprocess.check_output(
                "echo NO_PASS", shell=True).decode("utf-8")
        except subprocess.CalledProcessError:
            gdpass = ""
        d = dict(
            gdrive_str=gdrive_str.replace(
                "CONTENT_CREDENTIAL", "" if os.getenv(
                    "CI") == "true" else gdpass
            ),
            rvs_pty_config_str=rvs_pty_config_str,
            rvs_str=rvs_str,
            runner_src=runner_src,
            AMQPURL=AMQPURL.string(),
            size=size,
            network=net,
            seed=seed,
            gdrive_enable=gdrive_enable,
            port=port,
            phase=phase,
            server=os.environ.get("server", "vtool.duckdns.org"),
        )
        ss = s.safe_substitute(d)

        with open(os.path.join(path, "main.py"), "w") as jf:
            jf.write(ss)

    def run_local(self, path):
        return subprocess.run("python " + os.path.join(path, "main.py"), shell=True)

    def create_runner(self, config, seed="2020", script=True, from_template=True):
        """
        config will be size and model right now
        """
        size = config["size"]
        net = config["network"]
        name = net.replace("_", "-") + "-" + str(size)
        AMQPURL = config["AMQPURL"]
        port = config["port"]
        phase = config.get("phase","dev")

        path = os.path.join(self.tmp_path, name)
        shutil.copytree(self.template_path, path)

        if from_template:
            self._change_kernel_meta_info(path, None, script)
            self._change_main_py(path, size, net, AMQPURL, seed, port, phase=phase)
        else:
            self._change_kernel_meta_info(
                path, self.title_prefix + " " + name, script)
            self._change_main_py(path, size, net, AMQPURL, seed, port, phase=phase)

        if not script:
            subprocess.run(
                ("jupytext --to notebook " + os.path.join(path, "main.py")).split()
            )

        self.runners.append(path)

        return path

if __name__ == "__main__":
    port = sys.argv[1]
    assert int(port) >= 0
    phase = sys.argv[2]
    logger.debug(f"Paramters for creating runner: {sys.argv}")
    tmp_path = '.r'

    subprocess.run(f"rm -rf {tmp_path}", shell=True, check=True)
    coordinator = Coordinator(tmp_path, "Test Runner")
    config = {"phase": phase, "port":port, "size": 384, "network": "intercept", "AMQPURL": AMQPURL()}
    path = coordinator.create_runner(config, 19999, False)

    if os.getenv("CI") != "true":
        ret = coordinator.push(path)  # just push first
        assert ret.returncode == 0
