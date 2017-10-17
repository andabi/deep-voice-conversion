#!/usr/bin/env bash

gpu=$1
case=$2
logdir="/data/private/vc/logdir_$case/train1"

echo "gpu=$gpu"
echo "logdir=$logdir"

CUDA_VISIBLE_DEVICES=$gpu nohup python train1.py $logdir >> $case.out &
tail -f $case.out