#!/usr/bin/env bash

if [ $# -lt 3 ]
  then
    echo "insufficient arguments. case1=$1, case2=$2, gpu=$3"
    exit 0
fi

case1=$1
case2=$2
gpu=$3

echo "case1=$case1"
echo "case2=$case2"
echo "gpu=$gpu"

if [ ! -d "out" ]
  then
    mkdir out
fi

CUDA_VISIBLE_DEVICES=$gpu nohup python train2.py $case1 $case2 >> out/$case2.out &
tail -f out/$case2.out