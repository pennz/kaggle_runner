#!/bin/bash -x
# trap ctrl-c and call ctrl_c()
PS4='L${LINENO}: '

trap ctrl_c INT
NC=ncat

function ctrl_c() {
    echo >&2 "** Trapped CTRL-C"
}

getNewPort() {
    serverNodes=$1
    lastServer=$(tail -n 1 $serverNodes)
    if [ $? -eq 0 -a x"$lastServer" != x ]; then
        newNode=$((lastServer + 1))
    else
        newNode=9000
    fi
    echo $newNode >>$serverNodes
    echo $newNode
}

connect() {
    local newPort=$1

    pgrep -f "lp $newPort" && return 1 # port used
    ~/bin/upnp-add-port $newPort &     # port forward, rvs will connect to this port
    listen_pid=$!

    tmux new-window -t rvsConnector "stty raw -echo && { while true; do $NC -vvklp $newPort ; echo \"Disconnected, will re-listen again\"; sleep 1; done }"
    #ncat 127.1 $newPort

    #tmux new-window -t rvsConnector "stty raw -echo && { $NC 127.1 $newPort </dev/tty ; }"

    # tcpserver waits for connections from TCP clients. For each connection, it
    # runs prog, with descriptor 0 reading from the network and descriptor 1 writing
    # to the network. It also sets up several environment variables.
    # so I need a program print the despriptor 0 content out and receive tty input
    # orignially, use ncat -lp 9000, so RVS:48852 -> :25454 (tcpserver) -> ncat 9000 [0] from the RVS, and waiting input from [1]
    sleep 3
    wait $listen_pid
    ret=$?
    sed -i "/^$newPort\$/d" $2 1>/dev/null 2>&1 # ncat exit, then we delete in the booking
    return $ret
}

port=$(getNewPort serverNodes)
connect $port serverNodes

while [ ! $? -eq 0 ]; do
    echo "$port" >>serverNodes
    port=$((port + 1))
    connect $port serverNodes
done

echo -n "$port"

# so reverse shell server named to RSS
# so tcpserver instance named to TSins
#   RSS output -> TSins[0]
#   RSS input <- TSins[1]
# we let TSins[1] <- our keyboard input
# and TSins[0] -> our terminal screen
