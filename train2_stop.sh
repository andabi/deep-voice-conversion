#!/usr/bin/env bash

case=$1
logdir="/data/private/vc/logdir_$case/train2"

echo "logdir=$logdir"

for pid in $(ps -ef | grep "$logdir" | awk '{print $2}'); do kill $pid; done