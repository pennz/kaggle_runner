#!/bin/bash -x
# trap ctrl-c and call ctrl_c()
trap ctrl_c INT
NC=ncat

function ctrl_c() {
        echo "** Trapped CTRL-C"
}

addNewPort() {
  serverNodes=$1
  lastServer=$(tail -n 1 $serverNodes)
  if [ $? -eq 0 -a x"$lastServer" != x ]; then
    newNode=$((lastServer + 1))
  else
    newNode=9000
  fi
  echo $newNode >> $serverNodes
  echo $newNode
}

newPort=$(addNewPort serverNodes)
$NC -vlp $newPort
sed -i "/^$newPort\$/d" serverNodes 1>/dev/null 2>&1
# so reverse shell server named to RSS
# so tcpserver instance named to TSins
#   RSS output -> TSins[0]
#   RSS input <- TSins[1]
# we let TSins[1] <- our keyboard input
# and TSins[0] -> our terminal screen
