#!/usr/bin/env bash

if [ $# -lt 3 ]
  then
    echo "insufficient arguments. case1=$1, case2=$2, gpu=$3"
    exit 0
fi

case1=$1
case2=$2
gpu=$3
logdir1="/data/private/vc/logdir_$case1/train1"
logdir2="/data/private/vc/logdir_$case2/train2"

echo "logdir1=$logdir1"
echo "logdir2=$logdir2"
echo "gpu=$gpu"

if [ ! -d "out" ]
  then
    mkdir out
fi

CUDA_VISIBLE_DEVICES=$gpu nohup python train2.py $logdir1 $logdir2 >> out/$case2.out &
tail -f out/$case2.out