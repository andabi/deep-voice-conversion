#!/usr/bin/env bash

if [ $# -lt 2 ]
  then
    echo "insufficient arguments. case=$1, port=$2"
    exit 0
fi

case=$1
port=$2

sh scripts/tensorboard_stop.sh $case
sh scripts/tensorboard_start.sh $case $port