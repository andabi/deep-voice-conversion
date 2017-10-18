#!/usr/bin/env bash

port=$1
case=$2
logdir="/data/private/vc/logdir_$case"

echo "start. case=$case, port=$port"

nohup tensorboard --logdir=$logdir --port=$port --reload-interval=5 >> $case.tb.out &
