#!/bin/bash -x
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
pip install --upgrade pip &
pip3 install --upgrade pip &
conda install -y -c eumetsat expect & # https://askubuntu.com/questions/1047900/unbuffer-stopped-working-months-ago
apt update && apt install -y netcat nmap screen time locales >/dev/null 2>&1
apt install -y mosh iproute2 vim fish tig ctags htop tree pv tmux psmisc >/dev/null 2>&1 &

conda init bash
cat >>~/.bashrc <<EOF
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
        bash rvs.sh $SERVER $PORT 2>&1 &
    else
        screen -d -m bash -c "{ echo [REMOTE]: rvs log below.; bash -x rvs.sh $SERVER $PORT 2>&1; } | $NC --send-only --no-shutdown -w 120s -i $((3600 * 2))s $SERVER $CHECK_PORT"
    fi
fi &

python3 -m pip install pysnooper ipdb ripdb \
pytest-logger python_logging_rabbitmq coverage &
python3 -m pip install parse jupytext pydicom
#python3 -m pip install pyvim neovim msgpack==1.0.0 jedi &

SRC_WORK_FOLDER=/kaggle/working
[ -d ${SRC_WORK_FOLDER} ] || mkdir -p ${SRC_WORK_FOLDER}

cd ${SRC_WORK_FOLDER}

if [ -d ${REPO} ]; then rm -rf ${REPO}; fi

# get code

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
    #find . -maxdepth 1 -name ".??*" -o -name "??*" -type f | xargs -I{} mv {} $OLDPWD
    #find . -maxdepth 1 -name ".??*" -o -name "??*" -type d | xargs -I{} bash -x -c "mvdir {}  $OLDPWD"
    popd
    dp=$(mktemp)
    mkdir $dp && mv ${REPO} $dp && mv $dp/${REPO}/* mv $dp/${REPO}/.* .
fi

make install_dep >/dev/null &

USE_AMQP=true
export USE_AMQP

conda init bash
source ~/.bashrc
conda activate base

export PATH=$PWD/kaggle_runner/bin:$PATH
rvs.sh $SERVER $PORT >/dev/null & # just keep one rvs incase

if [ x"${PHASE}" = x"dev" ]; then
    export PS4='[Remote]: Line ${LINENO}: '
    (
        echo "MOSHing"
        make mosh
    ) &

    make toxic | if [ x$USE_AMQP = x"true" ]; then cat -; else $NC --send-only -w 120s -i $((60 * 5))s $SERVER $CHECK_PORT; fi &
    wait # not exit, when dev
fi

if [ x"${PHASE}" = x"data" ]; then
    make dataset
fi

if [ x"${PHASE}" = x"test" ]; then
    make test
fi

if [ x"${PHASE}" = x"pretrain" ]; then
    make mbd_pretrain
fi

if [ x"${PHASE}" = x"run" ]; then
    #pip install kaggle_runner
    make m & # just keep one rvs incase
    make toxic | if [ x"$USE_AMQP" = xtrue ]; then cat -; else $NC --send-only -w 120s -i $((60 * 5))s $SERVER $CHECK_PORT; fi
    # basically the reverse of the calling path
    pkill make &
    pkill -f "mosh" &
    pkill sleep &
    pkill -f "rvs.sh" &
    pkill ncat &
    # python main.py "$@"
fi
