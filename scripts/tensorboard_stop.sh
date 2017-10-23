#!/usr/bin/env bash

if [ $# -lt 1 ]
  then
    echo "insufficient arguments. case=$1"
    exit 0
fi

case=$1
logdir="logdir/$case"

echo "stop. case=$case"

for pid in $(ps -ef | grep tensorboard | grep "$logdir" | awk '{print $2}'); do kill $pid; done