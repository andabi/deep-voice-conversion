#!/usr/bin/env bash

if [ $# -lt 2 ]
  then
    echo "insufficient arguments. case=$1, gpu=$2"
    exit 0
fi

case=$1
gpu=$2
logdir="/data/private/vc/logdir_$case/train1"

echo "logdir=$logdir"
echo "gpu=$gpu"

if [ ! -d "out" ]
  then
    mkdir out
fi

CUDA_VISIBLE_DEVICES=$gpu nohup python train1.py $logdir >> out/$case.out &
tail -f out/$case.out