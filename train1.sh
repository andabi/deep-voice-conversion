#!/usr/bin/env bash

if [ $# -lt 2 ]
  then
    echo "insufficient arguments. case=$1, gpu=$2"
    exit 0
fi

case=$1
gpu=$2

echo "case=$case"
echo "gpu=$gpu"

if [ ! -d "out" ]
  then
    mkdir out
fi

CUDA_VISIBLE_DEVICES=$gpu nohup python train1.py $case >> out/$case.out &
tail -f out/$case.out