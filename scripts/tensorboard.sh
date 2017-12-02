#!/usr/bin/env bash

if [ $# -lt 2 ]
  then
    echo "insufficient arguments. case=$1, port=$2"
    exit 0
fi

case=$1
port=$2
base_logdir="./logdir"
logdir="$base_logdir/$case"

echo "start. case=$case, port=$port"

# Stop the existing first.
sh scripts/tensorboard_stop.sh $case

if [ ! -d "out" ]
  then
    mkdir out
fi

nohup tensorboard --logdir=$logdir --port=$port --reload-interval=5 >> out/$case.tb.out &
