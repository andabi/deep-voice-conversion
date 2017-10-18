#!/usr/bin/env bash

case=$1
logdir="/data/private/vc/logdir_$case"

echo "stop. case=$case"

for pid in $(ps -ef | grep tensorboard | grep "$logdir" | awk '{print $2}'); do kill $pid; done