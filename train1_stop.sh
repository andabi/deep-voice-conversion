#!/usr/bin/env bash

if [ $# -lt 1 ]
  then
    echo "insufficient arguments. case=$1"
    exit 0
fi

case=$1

echo "case=$case"

for pid in $(ps -ef | grep "python" | grep "train1" | grep "$case" | awk '{print $2}'); do kill $pid; done