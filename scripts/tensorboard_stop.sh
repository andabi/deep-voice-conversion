#!/usr/bin/env bash

if [ $# -lt 1 ]
  then
    echo "insufficient arguments. case=$1"
    exit 0
fi

case=$1
base_logdir="."
logdir="$base_logdir/logdir_$case"

echo "stop. case=$case"

for pid in $(ps -ef | grep tensorboard | grep "$logdir" | awk '{print $2}'); do kill $pid; done