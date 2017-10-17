#!/usr/bin/env bash

gpu=$1
case1=$2
case2=$3
logdir1="/data/private/vc/logdir_$case1/train1"
logdir2="/data/private/vc/logdir_$case2/train2"

echo "gpu=$gpu"
echo "logdir1=$logdir1"
echo "logdir2=$logdir2"

CUDA_VISIBLE_DEVICES=$gpu nohup python train2.py $logdir1 $logdir2 >> $case2.out &
tail -f $case2.out