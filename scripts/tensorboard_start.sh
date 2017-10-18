#!/usr/bin/env bash

if [ $# -lt 2 ]
  then
    echo "insufficient arguments. case=$1, port=$2"
    exit 0
fi

case=$1
port=$2
logdir="/data/private/vc/logdir_$case"

echo "start. case=$case, port=$port"

if [ ! -d "out" ]
  then
    mkdir out
fi

nohup tensorboard --logdir=$logdir --port=$port --reload-interval=5 >> out/$case.tb.out &
