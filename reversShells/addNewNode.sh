#!/bin/bash -x
# trap ctrl-c and call ctrl_c()
PS4='L${LINENO}: '

mosh=

if [ $# -gt 0 ]; then
    mosh=true
fi

trap ctrl_c INT
NC=ncat
trial_cnt=0

function ctrl_c() {
    echo >&2 "** Trapped CTRL-C"
}

getNewPort() {
    serverNodes=$1
    lastServer=$(tail -n 1 $serverNodes)
    [ $lastServer -gt 1000 ] || lastServer=8999
    if [ $? -eq 0 -a x"$lastServer" != x ]; then
        newNode=$((lastServer + 1))
    else
        newNode=9000
    fi
    unbuffer echo $newNode >>$serverNodes
    sync
    echo -n $newNode
}

mosh_connect() {
    local newPort=$1
    trial_cnt=$((trial_cnt + 1))

    pgrep -f "ncat.*p $newPort" >/dev/null && return 1 # port used
    ~/bin/upnp-add-port $newPort UDP >/dev/null 2>&1   # port forward, rvs will connect to this port
    ret=$?

    if [ $trial_cnt -gt 4 ]; then
        echo >&2 connection error thing
        exit 1 #
    fi

    (
        sleep 1
        bash pcc $newPort >/dev/null 2>&1
    ) &

    # echo "" # blank message, to activate? will make it fail?
    $NC -ulp $newPort

    sed -i "/^$newPort\$/d" $2 1>/dev/null 2>&1 # ncat exit, then we delete in the booking
    return $ret
}

connect() {
    local newPort=$1
    trial_cnt=$((trial_cnt + 1))

    pgrep -f "lp $newPort" >/dev/null && return 1 # port used
    ~/bin/upnp-add-port $newPort                  # port forward, rvs will connect to this port
    ret=$?

    if [ ! $ret -eq 0 ]; then
        exit $ret
    fi

    if [ $trial_cnt -gt 4 ]; then
        echo >&2 connection error thing
        exit 1 #
    fi

    tmux >/dev/null select-window -t rvsConnector:{end}
    tmux >/dev/null new-window -t rvsConnector:+1 -n "$(git show --no-patch --oneline)" "stty raw -echo && { while true; do $NC -vvlp $newPort ; echo \"Disconnected, will re-listen again\"; sleep 1; done }"

    # tcpserver waits for connections from TCP clients. For each connection, it
    # runs prog, with descriptor 0 reading from the network and descriptor 1 writing
    # to the network. It also sets up several environment variables.
    # so I need a program print the despriptor 0 content out and receive tty input
    # orignially, use ncat -lp 9000, so RVS:48852 -> :25454 (tcpserver) -> ncat 9000 [0] from the RVS, and waiting input from [1]
    sleep 3
    sed -i "/^$newPort\$/d" $2 1>/dev/null 2>&1 # ncat exit, then we delete in the booking

    return $ret
}

port=$(getNewPort serverNodes)

make_connect() {
    if [ -z $mosh ]; then
        connect $port serverNodes
    else
        mosh_connect $port serverNodes
    fi
}

make_connect

while [ ! $? -eq 0 ]; do
    port=$((port + 1))
    unbuffer echo "$port" >>serverNodes
    sync
    make_connect
done

if [ -z $mosh ]; then
    echo -n "$port"
fi

# so reverse shell server named to RSS
# so tcpserver instance named to TSins
#   RSS output -> TSins[0]
#   RSS input <- TSins[1]
# we let TSins[1] <- our keyboard input
# and TSins[0] -> our terminal screen
